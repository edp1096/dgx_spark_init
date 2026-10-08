from pathlib import Path
p=Path('/src/src/document_core/commands/object_ops/table.rs')
s=p.read_text()
old='let mut raw_ctrl_data = vec![0u8; 38];'
assert s.count(old)==2, 'Pinned rhwp table constructors changed; review the patch'
s=s.replace(old, '''// Complete CommonObjAttr: 36 fixed bytes, 4 page-break bytes,
        // and the mandatory 2-byte empty description length.
        let mut raw_ctrl_data = vec![0u8; 42];''')
p.write_text(s)

# Synthetic line segments are editing geometry, not saved page-break evidence.
# A newly appended paragraph has vpos=0; treating it as a stored reset creates
# an extra HWP page even though the paragraph fits on the current page.
p=Path('/src/src/renderer/typeset.rs')
s=p.read_text()
for old,new in [
 ('let curr_first_vpos = para.line_segs.first().map(|s| s.vertical_pos);',
  'let curr_first_vpos = para.line_segs.first().filter(|s| s.tag & 0x8000_0000 == 0).map(|s| s.vertical_pos);'),
 ('let prev_last_vpos = prev_para.line_segs.last().map(|s| s.vertical_pos);',
  'let prev_last_vpos = prev_para.line_segs.last().filter(|s| s.tag & 0x8000_0000 == 0).map(|s| s.vertical_pos);')]:
 old='\n                '+old
 new='\n                '+new
 assert s.count(old)==1, 'Pinned pagination guard changed; review the patch'
 s=s.replace(old,new)
p.write_text(s)

# Treat the verified sparse checkout as a source archive. Cargo 1.93 fingerprint
# scanning cannot traverse Git entries whose blobs were intentionally omitted.
import shutil
shutil.rmtree("/src/.git")
