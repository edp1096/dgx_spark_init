// Build-time correction for Excelize v2.11.0 OR returning text on its numeric true branch.
package main

import (
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	if len(os.Args) != 2 {
		panic("output directory required")
	}
	out := os.Args[1]
	data, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "github.com/xuri/excelize/v2").Output()
	if err != nil {
		panic(err)
	}
	source := strings.TrimSpace(string(data))
	err = filepath.WalkDir(source, func(p string, d fs.DirEntry, e error) error {
		if e != nil {
			return e
		}
		rel, e := filepath.Rel(source, p)
		if e != nil {
			return e
		}
		target := filepath.Join(out, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0755)
		}
		b, e := os.ReadFile(p)
		if e != nil {
			return e
		}
		return os.WriteFile(target, b, 0644)
	})
	if err != nil {
		panic(err)
	}
	target := filepath.Join(out, "calc.go")
	data, err = os.ReadFile(target)
	if err != nil {
		panic(err)
	}
	before := "return newStringFormulaArg(strings.ToUpper(strconv.FormatBool(or)))"
	if strings.Count(string(data), before) != 1 {
		panic("Excelize OR source changed; review the patch")
	}
	patched := strings.Replace(string(data), before, "return newBoolFormulaArg(or)", 1)
	andStart := strings.Index(patched, "func (fn *formulaFuncs) AND(")
	andEnd := strings.Index(patched[andStart:], "// FALSE function") + andStart
	andBody := patched[andStart:andEnd]
	andBefore := "return newStringFormulaArg(token.String)"
	if strings.Count(andBody, andBefore) != 1 {
		panic("Excelize AND source changed; review the patch")
	}
	andBody = strings.Replace(andBody, andBefore, "return newBoolFormulaArg(false)", 1)
	patched = patched[:andStart] + andBody + patched[andEnd:]

	if err = os.WriteFile(target, []byte(patched), 0644); err != nil {
		panic(err)
	}
	command := exec.Command("go", "mod", "edit", "-replace=github.com/xuri/excelize/v2="+out)
	if output, err := command.CombinedOutput(); err != nil {
		panic(string(output))
	}
	fmt.Println("Applied Excelize v2.11.0 OR boolean correction")
}
