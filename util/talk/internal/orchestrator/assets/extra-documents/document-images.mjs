export function validateImages(sections){
 let count=0,total=0;
 for(const section of sections||[])for(const item of section.images||[]){
  if(!item||typeof item!=='object'||Array.isArray(item)||Object.keys(item).some(k=>!['data','width_px','height_px','width_cm','caption'].includes(k)))throw Error('Invalid document image fields');
  if(++count>6)throw Error('At most 6 document images');
  if(typeof item.data!=='string'||item.data.length>7*1024*1024||!/^[A-Za-z0-9+/]*={0,2}$/.test(item.data))throw Error('Image must be base64 PNG');
  const bytes=Buffer.from(item.data,'base64');total+=bytes.length;
  if(total>8*1024*1024||bytes.length<33||!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))||bytes.toString('ascii',12,16)!=='IHDR')throw Error('Invalid PNG image or image budget exceeded');
  const w=bytes.readUInt32BE(16),h=bytes.readUInt32BE(20);
  if(w<1||h<1||w>1024||h>1024||item.width_px!==w||item.height_px!==h)throw Error('PNG dimensions must match and be at most 1024 px');
  if(!Number.isFinite(item.width_cm)||item.width_cm<1||item.width_cm>16)throw Error('Image width must be 1–16 cm');
  if(item.caption!==undefined&&(typeof item.caption!=='string'||item.caption.length>240||item.caption.includes('\0')))throw Error('Invalid image caption');
 }
}
export function imageSize(item){
 let width=item.width_cm*72/2.54,height=width*item.height_px/item.width_px;
 if(height>20*72/2.54){width*=20*72/2.54/height;height=20*72/2.54;}
 return {width,height};
}
