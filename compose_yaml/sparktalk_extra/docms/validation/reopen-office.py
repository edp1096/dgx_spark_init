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
import zipfile

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
        with zipfile.ZipFile(source) as archive:
            names=archive.namelist()
            if ext=='docx':
                result['table_count']=document.getTextTables().getCount()
                if b'TOC ' in archive.read('word/document.xml'):
                    result['toc_count']=document.getDocumentIndexes().getCount()
                    if result['toc_count']<1:raise RuntimeError('TOC did not import as a real index')
            elif ext=='pptx':
                expected=sum(1 for n in names if n.startswith('ppt/slides/slide') and n.endswith('.xml') and '/_rels/' not in n)
                result['slide_count']=document.getDrawPages().getCount()
                if result['slide_count']!=expected:raise RuntimeError('Slide count changed on import')
            elif ext=='xlsx':
                chart_count=sum(document.Sheets.getByIndex(i).getCharts().getCount() for i in range(document.Sheets.getCount()))
                pivot_count=sum(document.Sheets.getByIndex(i).getDataPilotTables().getCount() for i in range(document.Sheets.getCount()))
                result.update(chart_count=chart_count,pivot_count=pivot_count)
                if chart_count!=sum(1 for n in names if n.startswith('xl/charts/chart') and n.endswith('.xml')):raise RuntimeError('Chart lost on import')
                if pivot_count!=sum(1 for n in names if n.startswith('xl/pivotTables/pivotTable') and n.endswith('.xml')):raise RuntimeError('Pivot lost on import')

        if ext=='xlsx':
            document.calculateAll()
            controller=document.getCurrentController();frozen={}
            for name in document.Sheets.getElementNames():
                sheet=document.Sheets.getByName(name)
                if not sheet.IsVisible:continue
                controller.setActiveSheet(sheet)
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
