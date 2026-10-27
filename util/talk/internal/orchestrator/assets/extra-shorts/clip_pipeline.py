"""Reference edit -> inspect/repair -> H3 clip -> temporal inspect/repair.

Every completed step is a fingerprinted artifact. Retries keep the original
intent and bounded quality budgets; transport errors never consume a new seed.
"""
import base64
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Literal

from PIL import Image, ImageChops, ImageDraw, ImageStat
from pydantic import BaseModel, ConfigDict, Field, model_validator

import clip_store as store


class Region(BaseModel):
    model_config = ConfigDict(extra='forbid')
    label: str = Field(min_length=1,max_length=100)
    box: list[int] = Field(min_length=4,max_length=4)

    @model_validator(mode='after')
    def bounds(self):
        if not (0 <= self.box[0] < self.box[2] <= 1000 and 0 <= self.box[1] < self.box[3] <= 1000):
            raise ValueError('box must be [left,top,right,bottom] in normalized 0..1000 coordinates')
        return self


class Plan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    title: str = Field(min_length=1,max_length=100)
    scope: Literal['image_to_video','unsupported']
    unsupported_reason: str = Field(default='',max_length=1000)
    edit_required: bool
    image_prompt: str = Field(min_length=1,max_length=6000)
    motion_prompt: str = Field(min_length=1,max_length=6000)
    requested_motion: str = Field(min_length=1,max_length=1000)
    preserve: list[str] = Field(min_length=1,max_length=12)
    style: str = Field(min_length=1,max_length=500)
    audio: Literal['generated','mute'] = 'generated'
    inspect_regions: list[Region] = Field(default_factory=list,max_length=4)

    @model_validator(mode='after')
    def prompt_budget(self):
        preserve='; '.join(self.preserve)
        if len(preserve)>1500 or len(self.motion_prompt+self.requested_motion+preserve)>5000 or len(self.image_prompt+preserve)>5000:
            raise ValueError('Keep image and motion instructions concise: each prompt plus preserved elements must fit 5000 characters')
        return self


class Issue(BaseModel):
    model_config = ConfigDict(extra='forbid')
    category: Literal['hand','face','anatomy','style','composition','subject_count','motion','temporal','technical','other']
    severity: Literal['major','minor']
    evidence: str = Field(min_length=8,max_length=1200)
    target: Literal['image','video']
    region: Region | None = None
    times: list[float] = Field(default_factory=list,max_length=8)


class Review(BaseModel):
    model_config = ConfigDict(extra='forbid')
    verdict: Literal['pass','revise','uncertain']
    summary: str = Field(min_length=5,max_length=1500)
    issues: list[Issue] = Field(default_factory=list,max_length=8)
    repair_prompt: str = Field(default='',max_length=2500)

    @model_validator(mode='after')
    def consistent(self):
        if self.verdict=='pass' and any(i.severity=='major' for i in self.issues):
            raise ValueError('A major issue cannot pass')
        if self.verdict!='pass' and (not self.issues or not self.repair_prompt.strip()):
            raise ValueError('A failed or uncertain review needs concrete evidence and a repair/reinspection instruction')
        return self


class MotionRevision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    motion_prompt: str = Field(min_length=20,max_length=4000)
    changes: str = Field(min_length=5,max_length=1200)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def image_url(path, max_edge=960):
    with Image.open(path) as picture:
        picture=picture.convert('RGB');picture.thumbnail((max_edge,max_edge))
        result=io.BytesIO();picture.save(result,format='JPEG',quality=92)
    return 'data:image/jpeg;base64,'+base64.b64encode(result.getvalue()).decode()


def original_url(path):
    return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()


def pixel_box(region, size):
    left,top,right,bottom=region['box'];width,height=size
    return [max(0,math.floor(left*width/1000)),max(0,math.floor(top*height/1000)),
            min(width,math.ceil(right*width/1000)),min(height,math.ceil(bottom*height/1000))]


def crop_sheet(source, candidate, regions, output):
    if not regions:
        return None
    selected=regions[:4]
    sheet=Image.new('RGB',(768,416*len(selected)),'#171717');draw=ImageDraw.Draw(sheet)
    for row,region in enumerate(selected):
        for column,path in enumerate((source,candidate)):
            with Image.open(path) as im:
                crop=im.convert('RGB').crop(pixel_box(region,im.size));crop.thumbnail((384,384))
                x,y=column*384,row*416
                sheet.paste(crop,(x+(384-crop.width)//2,y+32))
                draw.text((x+8,y+8),f"{'ORIGINAL' if column==0 else 'CANDIDATE'} / {row+1}",fill='white')
    sheet.save(output,quality=93)
    return image_url(output,1536)


def author(bridge,project,job_id,phase,data,images):
    return bridge('author',{'session_id':project['session_id'],'job_id':job_id,
                           'phase':phase,'data':data,'images':images})


def structured(bridge,project,job_id,phase,data,images,schema,path):
    expected=fingerprint({'phase':phase,'data':data,'images':[hashlib.sha256(x.encode()).hexdigest() for x in images]})
    cached=read(path)
    if cached and cached.get('input_fingerprint')==expected:
        return schema.model_validate(cached['result']).model_dump()
    feedback=''
    for attempt in range(2):
        raw=author(bridge,project,job_id,phase,{**data,'schema':schema.model_json_schema(),'validation_feedback':feedback},images)
        store.atomic_json(path.with_name(path.stem+f'-raw-{attempt}.json'),raw)
        try:
            result=schema.model_validate(raw).model_dump()
            store.atomic_json(path,{'input_fingerprint':expected,'result':result})
            return result
        except ValueError as error:
            feedback=str(error)[:3000]
    raise ValueError('모델 응답 형식을 확인하지 못했습니다: '+feedback)


def image_review(bridge,project,job_id,root,plan,path,label):
    regions=plan['inspect_regions']
    evidence=[image_url(root/'source.png'),image_url(path)]
    crops=crop_sheet(root/'source.png',path,regions,root/(label+'-crops.jpg'))
    if crops:evidence.append(crops)
    data={'brief':project['brief'],'user_request':project.get('user_request',''),'plan':plan,'evidence_order':['original source','candidate','paired region crops'],
          'regions':regions,'scope':'Compare full composition AND enlarged visible hands/faces. Preserve intentional stylization. An occluded finger is not automatically a missing finger. Do not invent invisible defects. Judge the actual candidate, not the requested improvement.'}
    review=structured(bridge,project,job_id,'image_review',data,evidence,Review,root/(label+'-review.json'))
    if review['verdict']=='uncertain':
        focus=[i['region'] for i in review['issues'] if i.get('region')]
        focused=crop_sheet(root/'source.png',path,focus,root/(label+'-focus.jpg'))
        if focused:
            review=structured(bridge,project,job_id,'image_review',{**data,'previous_uncertain_review':review,'scope':'Resolve the specified uncertainty from the enlarged regions. Return pass only when the actual evidence supports it.'},
                              [*evidence[:2],focused],Review,root/(label+'-focused-review.json'))
    return review


def generate_image(bridge,project,job_id,root,plan,index,previous,feedback,prior_categories):
    operation='identity_edit';source=root/'source.png';box=None
    prompt=plan['image_prompt']
    if feedback:
        prompt+='\nRequired correction: '+feedback['repair_prompt']+'\nKeep the requested style and: '+'; '.join(plan['preserve'])
        local=[i for i in feedback['issues'] if i['severity']=='major' and i.get('region') and i['target']=='image']
        categories={i['category'] for i in feedback['issues'] if i['severity']=='major'}
        # Repeated local failure switches back to a fresh edit of the original.
        if previous and len(local)==1 and not categories.intersection(prior_categories):
            operation='inpaint';source=previous
            with Image.open(source) as image:box=pixel_box(local[0]['region'],image.size)
    request={'session_id':project['session_id'],'job_id':job_id,'prompt':prompt,
             'operation':operation,'seed':project['seed']+index,'source_image':original_url(source)}
    if box:request['mask_box']=box
    result_path=root/f'image-{index}.png';receipt_path=root/f'image-{index}-generation.json'
    key=fingerprint({**request,'source_image':digest(source)})
    receipt=read(receipt_path)
    if receipt and receipt.get('input_fingerprint')==key and result_path.exists() and digest(result_path)==receipt.get('sha256'):
        return result_path,receipt
    response=bridge('image',request)
    raw=base64.b64decode(response['data'],validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.format!='PNG' or image.width*image.height>16*1024**2:
            raise ValueError('잘못된 편집 이미지가 반환됐습니다')
        image.verify()
    temporary=result_path.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(result_path)
    receipt={'input_fingerprint':key,'sha256':digest(result_path),'source_sha256':digest(source),'source':source.name,
             'operation':operation,'prompt':prompt,'mask_box':box,'seed':request['seed']}
    if box:
        with Image.open(source) as before,Image.open(result_path) as after:
            if before.size!=after.size:
                raise ValueError('영역 편집이 원본 크기를 변경했습니다')
            difference=ImageChops.difference(before.convert('RGB'),after.convert('RGB'))
            difference.paste((0,0,0),tuple(box))
            if difference.getbbox():
                raise ValueError('영역 편집이 지정 영역 밖의 픽셀을 변경했습니다')
        receipt['outside_mask_identical']=True
    store.atomic_json(receipt_path,receipt)
    return result_path,receipt


def probe(path):
    raw=subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)],timeout=60)
    info=json.loads(raw);streams=[s for s in info['streams'] if s['codec_type']=='video']
    if len(streams)!=1:raise ValueError('영상 스트림이 올바르지 않습니다')
    video=streams[0];fps=video['avg_frame_rate'].split('/')
    duration=float(info['format']['duration']);frames=int(video.get('nb_read_frames',video.get('nb_frames',0)))
    if frames!=124 or abs(float(fps[0])/float(fps[1])-24)>.01 or abs(duration-124/24)>.12:
        raise ValueError('생성 영상의 길이·프레임 수가 요청과 다릅니다')
    return {'width':video['width'],'height':video['height'],'frames':frames,'fps':24,'duration':duration,
            'has_audio':any(s['codec_type']=='audio' for s in info['streams'])}


def frame(path,time,output):
    subprocess.run(['ffmpeg','-y','-v','error','-ss',str(time),'-i',str(path),'-frames:v','1',str(output)],check=True,timeout=45)


def montage(files,times,output,columns=3,cell=(320,480)):
    width,height=cell
    sheet=Image.new('RGB',(width*columns,(height+28)*math.ceil(len(files)/columns)),'#181818')
    draw=ImageDraw.Draw(sheet)
    for i,(path,time) in enumerate(zip(files,times)):
        with Image.open(path) as picture:
            picture=picture.convert('RGB');picture.thumbnail(cell)
            x,y=(i%columns)*width,(i//columns)*(height+28)
            sheet.paste(picture,(x+(width-picture.width)//2,y+28))
            draw.text((x+8,y+8),f'{time:.3f}s',fill='white')
    sheet.save(output,quality=93)
    return output


def video_evidence(root,path,label):
    destination=root/label;destination.mkdir(exist_ok=True)
    times=[0,.75,1.75,2.75,3.75,5.125]
    files=[]
    for i,t in enumerate(times):
        target=destination/f'overview-{i}.png';frame(path,t,target);files.append(target)
    overview=montage(files,times,destination/'overview.jpg')
    # Locate motion on sampled frames, then inspect real adjacent frames at
    # native 24fps rather than claiming temporal checks from isolated stills.
    scores=[]
    for i in range(1,len(files)):
        with Image.open(files[i-1]) as first,Image.open(files[i]) as second:
            diff=ImageChops.difference(first.convert('L').resize((96,96)),second.convert('L').resize((96,96)))
            scores.append((ImageStat.Stat(diff).mean[0],i))
    peak=max(scores)[1]
    center=(times[peak-1]+times[peak])/2
    consecutive=[max(0,min(123/24,round(center*24)/24+(i-2)/24)) for i in range(5)]
    details=[]
    for i,t in enumerate(consecutive):
        target=destination/f'temporal-{i}.png';frame(path,t,target);details.append(target)
    temporal=montage(details,consecutive,destination/'temporal.jpg',columns=5,cell=(256,384))
    return {'overview':overview,'temporal':temporal,'times':times,'consecutive_times':consecutive}


def video_review(bridge,project,job_id,root,plan,path,reference,index):
    evidence=video_evidence(root,path,f'video-{index}-evidence')
    images=[image_url(reference),image_url(evidence['overview'],1800),image_url(evidence['temporal'],1600)]
    data={'brief':project['brief'],'user_request':project.get('user_request',''),'plan':plan,'evidence_order':['accepted starting image','six timestamped video frames','five adjacent 24fps frames near strong motion'],
          'frame_times':evidence['times'],'consecutive_times':evidence['consecutive_times'],
          'scope':'Judge requested action, style retention, subject count, hands/faces, anatomy and consistency across the supplied frames. Do not judge soundtrack from images. Distinguish a defect already visible in the starting image (target image) from one introduced during motion (target video). A series of still frames is limited evidence; do not claim exact lip sync or exhaustive frame inspection.'}
    review=structured(bridge,project,job_id,'video_review',data,images,Review,root/f'video-{index}-review.json')
    if review['verdict']=='uncertain':
        focus=[issue for issue in review['issues'] if issue.get('region') and issue.get('times')]
        if focus:
            issue=focus[0];paths=[];times=[]
            for i,t in enumerate(issue['times'][:4]):
                if t<0 or t>123/24:continue
                full=root/f'video-{index}-focus-{i}-full.png';frame(path,t,full)
                with Image.open(full) as im:
                    cropped=im.crop(pixel_box(issue['region'],im.size));target=root/f'video-{index}-focus-{i}.png';cropped.save(target)
                paths.append(target);times.append(t)
            if paths:
                detail=montage(paths,times,root/f'video-{index}-focus.jpg',columns=len(paths),cell=(384,384))
                review=structured(bridge,project,job_id,'video_review',{**data,'previous_uncertain_review':review,'focus_region':issue['region']},
                                  [*images,image_url(detail,1600)],Review,root/f'video-{index}-focused-review.json')
    return review,evidence


def produce(job_id,bridge):
    job=store.get(job_id);project=job['project'];root=store.folder(job_id)
    if digest(root/'source.png')!=project['source_sha256']:
        raise ValueError('원본 이미지가 제작 시작 이후 변경됐습니다')
    state=read(root/'state.json',{'version':1,'images':[],'videos':[],'selected_image':None,'image_feedback':None,'video_feedback':None})
    def stage(label,progress):
        current=store.get(job_id)
        if current['canceled']:raise RuntimeError('제작이 중지됐습니다')
        store.update(job_id,'running',label,max(current['progress'],progress))
    def checkpoint():store.atomic_json(root/'state.json',state)
    stage('사진과 요청을 보고 연출 계획 구성',5)
    plan=structured(bridge,project,job_id,'plan',{'brief':project['brief'],'user_request':project.get('user_request',''),'duration_seconds':124/24,'resolution':project['resolution'],
        'scope':'Style/edit a single existing image, then animate it. Face/person swaps and editing an existing video are out of scope. Preserve requested motion rather than removing it to simplify production.'},
        [image_url(root/'source.png')],Plan,root/'plan-response.json')
    store.atomic_json(root/'plan.json',plan)
    if plan['scope']!='image_to_video':raise ValueError(plan['unsupported_reason'] or '이 제작 흐름은 사진 편집 후 영상화만 처리합니다')
    def finish():
        stage('검수를 통과한 결과 저장',95)
        accepted=state['videos'][-1]
        candidate=root/state['accepted_video'];reference=root/state['selected_image']
        if accepted['review']['verdict']!='pass' or digest(candidate)!=accepted['generation']['sha256']:
            raise ValueError('검수한 영상과 저장된 파일이 다릅니다')
        if digest(reference)!=accepted['generation']['first_frame_sha256']:
            raise ValueError('검수한 시작 이미지가 변경됐습니다')
        if plan['audio']=='mute':
            subprocess.run(['ffmpeg','-y','-v','error','-i',str(candidate),'-map','0:v:0','-c:v','copy','-an',str(root/'final.mp4')],check=True,timeout=60)
        else:shutil.copyfile(candidate,root/'final.mp4')
        shutil.copyfile(reference,root/'final-image.png')
        shutil.copyfile(root/f"video-{len(state['videos'])-1}-evidence"/'overview.jpg',root/'preview-sheet.jpg')
        report={'passed':True,'plan':plan,'source_image_id':project['source_image_id'],'source_sha256':project['source_sha256'],
                'final_image_sha256':digest(reference),'final_video_sha256':digest(root/'final.mp4'),
                'metadata':probe(root/'final.mp4'),'image_attempts':state['images'],'video_attempts':state['videos'],
                'review_scope':'Local VLM compared full images and enlarged regions; inspected six video samples and five adjacent frames near the strongest sampled visual change. This is not every-frame inspection.',
                'retries':{'image':max(0,len(state['images'])-1),'video':max(0,len(state['videos'])-1)}}
        store.atomic_json(root/'report.json',report)
        stage('검사·수정 완료',100)
        store.update(job_id,'completed','검사·수정 완료',100)
    if state.get('accepted_video'):
        finish();return
    while True:
        while not state['selected_image']:
            index=len(state['images'])
            if index>=3:raise ValueError('이미지 수정 2회를 마쳤지만 검수를 통과하지 못했습니다. 실패 원인을 검사 기록에 보존했습니다.')
            stage(f'사진 스타일 편집·수정 {index+1}/3',15+index*7)
            previous=root/state['images'][-1]['file'] if state['images'] else None
            prior_categories={i['category'] for item in state['images'][:-1] for i in item['review']['issues'] if i['severity']=='major'}
            if index==0 and not plan['edit_required']:
                candidate=root/'source.png';receipt={'operation':'reuse_original','sha256':digest(candidate)}
            else:
                candidate,receipt=generate_image(bridge,project,job_id,root,plan,index,previous,state['image_feedback'],prior_categories)
            stage('원본과 편집 결과의 손·얼굴·구도 검사',35)
            review=image_review(bridge,project,job_id,root,plan,candidate,f'image-{index}')
            state['images'].append({'file':candidate.name,'generation':receipt,'review':review})
            if review['verdict']=='pass':
                state['selected_image']=candidate.name;state['image_feedback']=None
            else:state['image_feedback']=review
            checkpoint()
        reference=root/state['selected_image']
        if digest(reference)!=state['images'][-1]['generation']['sha256']:
            raise ValueError('검수한 이미지가 변경됐습니다')
        index=len(state['videos'])
        if index>=3:raise ValueError('영상 재생성 2회를 마쳤지만 검수를 통과하지 못했습니다. 실패 결과를 완성본으로 전달하지 않았습니다.')
        stage(f'검수한 사진으로 영상 생성 {index+1}/3',48+index*7)
        prompt=plan['motion_prompt']
        if state['video_feedback']:
            stage('검수 결과를 반영해 움직임 지시 재작성',52+index*7)
            evidence=[image_url(reference)]
            previous=root/f'video-{index-1}-evidence'/'overview.jpg'
            if previous.exists():evidence.append(image_url(previous,1800))
            revision=structured(bridge,project,job_id,'motion_repair',{
                'brief':project['brief'],'user_request':project.get('user_request',''),'plan':plan,
                'failed_review':state['video_feedback'],'previous_prompt':state['videos'][-1]['generation']['prompt'],
                'scope':'The first image is already edited and accepted. Rewrite only the motion instruction. Describe the desired action and stable appearance positively; do not repeat unrequested artifact names or ask to restyle the photo again. Preserve every requested action.'},
                evidence,MotionRevision,root/f'video-{index}-motion-repair.json')
            prompt=revision['motion_prompt']
            stage(f'수정한 움직임으로 영상 생성 {index+1}/3',55+index*7)
        prompt+='\nRequired action: '+plan['requested_motion']+'\nPreserve the starting image style and: '+'; '.join(plan['preserve'])
        request={'session_id':project['session_id'],'job_id':job_id,'prompt':prompt,'seed':project['seed']+8+index,
                 'resolution':project['resolution'],'first_frame':original_url(reference)}
        candidate=root/f'video-{index}.mp4';receipt_path=root/f'video-{index}-generation.json'
        key=fingerprint({**request,'first_frame':digest(reference)})
        receipt=read(receipt_path)
        if not receipt or receipt.get('input_fingerprint')!=key or not candidate.exists() or digest(candidate)!=receipt.get('sha256'):
            response=bridge('video',request)
            raw=base64.b64decode(response['data'],validate=True)
            temporary=candidate.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(candidate)
            metadata=probe(candidate)
            if metadata['width']!=response['width'] or metadata['height']!=response['height']:
                raise ValueError('영상의 실제 크기와 모델 응답이 다릅니다')
            receipt={'input_fingerprint':key,'sha256':digest(candidate),'first_frame_sha256':digest(reference),
                     'first_frame_file':reference.name,'seed':request['seed'],'prompt':prompt,'metadata':metadata}
            store.atomic_json(receipt_path,receipt)
        stage('최종 영상의 움직임·손·얼굴·스타일 검사',78)
        review,evidence=video_review(bridge,project,job_id,root,plan,candidate,reference,index)
        state['videos'].append({'file':candidate.name,'generation':receipt,'review':review,
                               'evidence':{'times':evidence['times'],'consecutive_times':evidence['consecutive_times']}})
        if review['verdict']=='pass':
            state['accepted_video']=candidate.name;checkpoint();finish()
            return
        image_issues=[i for i in review['issues'] if i['target']=='image' and i['severity']=='major']
        if image_issues:
            state['image_feedback']={**review,'issues':image_issues};state['selected_image']=None
        state['video_feedback']=review
        checkpoint()
