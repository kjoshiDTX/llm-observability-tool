package adapters

import (
	"bytes"
	"io"
	"strings"
)

func bytesReader(b []byte) io.Reader { return bytes.NewReader(b) }

func hasSuffix(s, suffix string) bool { return strings.HasSuffix(s, suffix) }
