"""Independent format checks, not a substitute for Hancom Office execution."""
import argparse
import json
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile


def local(element):
    return element.tag.rsplit('}', 1)[-1]


def check_hwpx(file):
    with zipfile.ZipFile(file) as archive:
        assert archive.testzip() is None, 'ZIP CRC failure'
        names = set(archive.namelist())
        assert archive.read('mimetype').decode().strip() == 'application/hwp+zip'
        for name in names:
            assert '..' not in pathlib.PurePosixPath(name).parts and not name.startswith('/')
            if name.endswith(('.xml', '.hpf', '.rdf')):
                ET.fromstring(archive.read(name))
        package = ET.fromstring(archive.read('Contents/content.hpf'))
        items = [e for e in package.iter() if local(e) == 'item']
        manifest = {e.attrib['id']: e.attrib['href'] for e in items}
        assert len(manifest) == len(items), 'Duplicate manifest IDs'
        assert all(href in names for href in manifest.values()), 'Missing manifest file'
        for e in package.iter():
            if local(e) == 'itemref':
                assert e.attrib['idref'] in manifest, 'Unresolved spine reference'
        header = ET.fromstring(archive.read('Contents/header.xml'))
        sections = [n for n in names if n.startswith('Contents/section') and n.endswith('.xml')]
        assert int(header.attrib['secCnt']) == len(sections), 'Section count mismatch'
        refs = {kind: {e.attrib['id'] for e in header.iter() if local(e) == kind}
                for kind in ['charPr', 'paraPr', 'style']}
        text = []
        tables = pictures = 0
        for name in sections:
            for e in ET.fromstring(archive.read(name)).iter():
                if local(e) == 't':
                    text.append(''.join(e.itertext()))
                if local(e) == 'tbl':
                    tables += 1
                    cells = [n for n in e.iter() if local(n) == 'tc']
                    assert len(cells) == 6, 'Sample table cells changed'
                if local(e) == 'img':
                    pictures += 1
                    item = e.attrib['binaryItemIDRef']
                    assert item in manifest, 'Unresolved image reference'
                    assert archive.read(manifest[item]).startswith(b'\x89PNG\r\n\x1a\n'), 'Invalid PNG'
                for attr, kind in [('charPrIDRef', 'charPr'), ('paraPrIDRef', 'paraPr'), ('styleIDRef', 'style')]:
                    if attr in e.attrib:
                        assert e.attrib[attr] in refs[kind], f'Unresolved {attr}'
        assert tables == 1 and pictures == 1, 'Missing table/image'
        body = ''.join(text)
        for expected in ['SparkTalk', '가나다라마바사', 'Sample 123']:
            assert expected in body, f'Missing text: {expected}'
        if file.stem == 'edited':
            assert '수정 검증' in body, 'Edit not persisted'
        # Keep a copy without the renderer-specific binary origin metadata.
        clean = file.with_name(file.stem + '-standard.hwpx')
        with zipfile.ZipFile(clean, 'w') as output:
            for item in archive.infolist():
                if not item.filename.startswith('META-INF/rhwp-'):
                    output.writestr(item, archive.read(item.filename))
        return {'status': 'passed', 'checks': 'ZIP/XML, manifest, style/image references, table cells, text/edit',
                'hancom_execution': 'not_available'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=pathlib.Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    report = {}
    command = pathlib.Path(sys.executable).parent / 'hwp5proc'
    for name in ['text-only.hwp', 'sample.hwp', 'edited.hwp']:
        process = subprocess.run([str(command), 'xml', '--no-validate-wellformed', str(directory / name)],
                                 capture_output=True, timeout=30)
        (directory / (name + '.xml')).write_bytes(process.stdout)
        (directory / (name + '.log')).write_bytes(process.stderr)
        passed = process.returncode == 0
        if passed:
            try:
                root = ET.fromstring(process.stdout)
                body = ''.join(root.itertext())
                expected = ['SparkTalk', '가나다라마바사', 'Sample 123']
                if name == 'edited.hwp': expected.append('수정 검증')
                passed = all(text in body for text in expected)
            except ET.ParseError:
                passed = False
        detail = next((line for line in process.stderr.decode(errors='replace').splitlines()
                       if 'ParseError' in line or 'Caused by:' in line), '')
        report[name] = {'status': 'passed' if passed else 'failed', 'parser': 'pyhwp', 'detail': detail}
    for name in ['sample.hwpx', 'edited.hwpx']:
        try:
            report[name] = check_hwpx(directory / name)
        except Exception as error:
            report[name] = {'status': 'failed', 'detail': str(error)}
    (directory / 'independent-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for name, result in report.items():
        print(name, result['status'], result.get('detail', ''))
    return int(any(result['status'] != 'passed' for result in report.values()))


if __name__ == '__main__':
    sys.exit(main())
