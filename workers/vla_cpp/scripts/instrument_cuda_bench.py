"""Record unsorted per-call latency without changing model inference."""
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
marker = '    std::printf("vla-bench: samples_ms=");'
anchor = '    std::sort(ms.begin(), ms.end());'
if marker not in source:
    if source.count(anchor) != 1:
        raise SystemExit('Expected exactly one sample-sort anchor')
    source = source.replace(anchor, '''    std::printf("vla-bench: samples_ms=");
    for (size_t i = 0; i < ms.size(); ++i)
        std::printf("%s%.6f", i ? "," : "", ms[i]);
    std::printf("\\n");
''' + anchor)
    path.write_text(source)
