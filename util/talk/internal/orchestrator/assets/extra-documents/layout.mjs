export const FONT='Noto Sans CJK KR';
export const COLOR={accent:'264D73',title:'18334D',muted:'68798B'};
export const SLIDE={width:960,height:540,title:{x:46.8,y:28.8,w:864,h:72,font:28},body:{x:57.6,y:118.8,w:849.6,h:360},footer:{x:820.8,y:507.6,w:86.4,h:14.4}};
export function slideFont(bullets){
 for(let size=20;size>=12;size--){const lines=bullets.reduce((n,t)=>n+t.split('\n').reduce((m,line)=>m+Math.max(1,Math.ceil([...line].reduce((w,c)=>w+(c.codePointAt(0)>255?1:0.6),0)/(SLIDE.body.w/size-2))),0),0);if(lines*size*1.35+Math.max(0,bullets.length-1)*12<=SLIDE.body.h)return size;}
 throw Error('Split long slide content into more slides');
}
