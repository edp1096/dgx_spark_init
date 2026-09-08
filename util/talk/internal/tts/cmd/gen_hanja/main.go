package main

import (
	"bufio"
	"compress/gzip"
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

func main() {
	input := flag.String("input", "", "path to Unicode Unihan_Readings.txt")
	output := flag.String("output", "", "path to generated gzip TSV")
	flag.Parse()
	if *input == "" || *output == "" {
		fmt.Fprintln(os.Stderr, "usage: gen_hanja -input Unihan_Readings.txt -output hanja_readings.tsv.gz")
		os.Exit(2)
	}

	in, err := os.Open(*input)
	check(err)
	defer in.Close()
	readings := make(map[int]string)
	scanner := bufio.NewScanner(in)
	for scanner.Scan() {
		fields := strings.Split(scanner.Text(), "\t")
		if len(fields) != 3 || fields[1] != "kHangul" || !strings.HasPrefix(fields[0], "U+") {
			continue
		}
		codepoint, parseErr := strconv.ParseInt(strings.TrimPrefix(fields[0], "U+"), 16, 32)
		check(parseErr)
		values := strings.Fields(fields[2])
		if len(values) == 0 {
			continue
		}
		reading := strings.SplitN(values[0], ":", 2)[0]
		if reading != "" {
			readings[int(codepoint)] = reading
		}
	}
	check(scanner.Err())

	keys := make([]int, 0, len(readings))
	for codepoint := range readings {
		keys = append(keys, codepoint)
	}
	sort.Ints(keys)
	out, err := os.Create(*output)
	check(err)
	gz := gzip.NewWriter(out)
	for _, codepoint := range keys {
		_, err = fmt.Fprintf(gz, "%X\t%s\n", codepoint, readings[codepoint])
		check(err)
	}
	check(gz.Close())
	check(out.Close())
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
