"""CPU admission and exact outside-mask compositing for native Qwen edits."""
import hashlib
import io
import math

from PIL import Image, ImageOps

from image_inputs import decode_keyframe


OPERATIONS = ('generate', 'identity_edit', 'reference_generate', 'inpaint')


def decode(value):
    raw, _ = decode_keyframe(value)
    with Image.open(io.BytesIO(raw)) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def png(image):
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


def working_image(image):
    scale = min(1.0, math.sqrt(1024 * 1024 / (image.width * image.height)))
    size = tuple(max(32, round(value * scale / 16) * 16) for value in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def image_size(value):
    try:
        width, height = map(int, value.split('x'))
    except (ValueError, AttributeError) as error:
        raise ValueError('size must be WIDTHxHEIGHT') from error
    if any(x < 256 or x > 1024 or x % 16 for x in (width, height)):
        raise ValueError('image dimensions must be 256..1024 in multiples of 16')
    return width, height


def prepare(body):
    operation = body.operation
    if operation not in OPERATIONS:
        raise ValueError('Unsupported image operation')
    if body.n != 1 or body.response_format != 'b64_json' or body.output_format != 'png':
        raise ValueError('Image generation returns one b64_json PNG')
    size = image_size(body.size)
    references = [working_image(decode(value)) for value in body.reference_images]
    source = decode(body.source_image or body.anypaint_image) if body.source_image or body.anypaint_image else None
    mask = None
    prompt = body.prompt.strip()
    if operation == 'generate':
        if source is not None or references or body.mask_box or body.anypaint_mask:
            raise ValueError('Use identity_edit or reference_generate with input images')
    elif operation == 'identity_edit':
        if source is not None:
            references.insert(0, working_image(source))
        if not references:
            raise ValueError('identity_edit requires source_image or reference_images')
        if body.mask_box or body.anypaint_mask:
            raise ValueError('Use inpaint for masked edits')
    elif operation == 'reference_generate':
        if source is not None or body.mask_box or body.anypaint_mask or not references:
            raise ValueError('reference_generate requires only reference_images')
    elif operation == 'inpaint':
        if source is None or (body.mask_box is None) == (body.anypaint_mask is None):
            raise ValueError('inpaint requires a source and exactly one mask_box or mask image')
        if body.mask_box is not None:
            box = body.mask_box
            if len(box) != 4 or not (0 <= box[0] < box[2] <= source.width and 0 <= box[1] < box[3] <= source.height):
                raise ValueError('mask_box must be [left,top,right,bottom] in source pixels')
            mask = Image.new('L', source.size)
            mask.paste(255, tuple(box))
        else:
            mask = decode(body.anypaint_mask).convert('L')
            if mask.size != source.size or mask.getextrema()[1] == 0:
                raise ValueError('mask must match source dimensions and contain an editable region')
        scaled = working_image(source)
        references = [scaled, mask.resize(scaled.size, Image.Resampling.NEAREST).convert('RGB'), *references]
        prompt = ('Image 1 is the source. Image 2 is an edit mask: white is editable, black must remain unchanged. '
                  'Repair only the white region, preserving the scene, rendering style and all other content. ' + prompt)
    if not prompt or len(references) > 4:
        raise ValueError('A nonempty prompt and at most four total reference inputs are required')
    raw = [png(image) for image in references]
    return {'prompt': prompt, 'references': raw, 'hashes': [hashlib.sha256(x).hexdigest() for x in raw],
            'source': source, 'mask': mask, 'size': size, 'operation': operation}


def finish(raw, prepared):
    with Image.open(io.BytesIO(raw)) as output:
        output = output.convert('RGB')
        if prepared['mask'] is not None:
            source = prepared['source']
            output = output.resize(source.size, Image.Resampling.LANCZOS)
            output = Image.composite(output, source, prepared['mask'])
        return png(output)
