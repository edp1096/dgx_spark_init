#!/usr/bin/env python3
"""Open in an actual LibreOffice controller, export PDF, then Save As OOXML.
Headless --convert-to drops window state (including frozen panes), even for a
separately generated openpyxl baseline. Never reconstruct that state in a test.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.document.MacroExecMode import NEVER_EXECUTE
from com.sun.star.document.UpdateDocMode import NO_UPDATE

source, destination = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
ext = source.suffix[1:]
filters = {'docx': ('Office Open XML Text','writer_pdf_Export'),
           'pptx': ('Impress MS PowerPoint 2007 XML','impress_pdf_Export'),
           'xlsx': ('Calc MS Excel 2007 XML','calc_pdf_Export')}
def prop(name,value):
    result=PropertyValue();result.Name=name;result.Value=value;return result

def url(path): return uno.systemPathToFileUrl(str(path))

with tempfile.TemporaryDirectory(prefix='office-gui-') as profile:
    pipe='sparktalk_verify_'+str(os.getpid())
    log=open(destination/'libreoffice.log','wb')
    process=subprocess.Popen(['libreoffice','-env:UserInstallation='+url(Path(profile)),
        '--nologo','--nodefault','--norestore','--nofirststartwizard',
        '--accept=pipe,name='+pipe+';urp;StarOffice.ServiceManager'],stdout=log,stderr=log)
    document=None;desktop=None
    try:
        local=uno.getComponentContext()
        resolver=local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver',local)
        deadline=time.monotonic()+20
        while True:
            try: context=resolver.resolve('uno:pipe,name='+pipe+';urp;StarOffice.ComponentContext');break
            except Exception:
                if time.monotonic()>deadline or process.poll() is not None:raise RuntimeError('LibreOffice controller did not start')
                time.sleep(.1)
        desktop=context.ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop',context)
        document=desktop.loadComponentFromURL(url(source),'_blank',0,
            (prop('Hidden',False),prop('ReadOnly',False),prop('MacroExecutionMode',NEVER_EXECUTE),prop('UpdateDocMode',NO_UPDATE)))
        if document is None:raise RuntimeError('LibreOffice could not open '+source.name)
        result={'source':source.name,'visible_controller':True}
        if ext=='xlsx':
            document.calculateAll()
            controller=document.getCurrentController();frozen={}
            for name in document.Sheets.getElementNames():
                controller.setActiveSheet(document.Sheets.getByName(name))
                frozen[name]=controller.hasFrozenPanes()
            result['frozen_on_open']=frozen
            if not all(frozen.values()):raise RuntimeError('Frozen header did not load in the real controller: '+str(frozen))
        document.storeToURL(url(destination/'document.pdf'),(prop('FilterName',filters[ext][1]),prop('Overwrite',True)))
        document.storeAsURL(url(destination/('document.'+ext)),(prop('FilterName',filters[ext][0]),prop('Overwrite',True)))
        (destination/'controller.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(ext+': visible-controller open/save passed')
    finally:
        if document is not None:document.close(True)
        if desktop is not None:desktop.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()
