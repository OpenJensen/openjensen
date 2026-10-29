"""Bounded byte-to-line framing shared by subprocess log collectors."""


class BoundedLines:
    def __init__(self, maximum=65536):
        self.maximum = maximum
        self.pending = bytearray()
        self.dropping = False
        self.skip_lf = False

    def feed(self, chunk):
        lines = []
        if self.skip_lf and chunk:
            if chunk.startswith(b"\n"):
                chunk = chunk[1:]
            self.skip_lf = False
        for part in chunk.splitlines(keepends=True):
            if not self.dropping:
                if len(self.pending) + len(part) > self.maximum:
                    self.pending.clear()
                    self.dropping = True
                else:
                    self.pending.extend(part)
            if part.endswith((b"\n", b"\r")):
                self.skip_lf = part.endswith(b"\r")
                lines.append(self._take())
        return lines

    def _take(self):
        value = (
            "[oversized worker line omitted]"
            if self.dropping
            else bytes(self.pending).decode("utf-8", errors="replace").rstrip("\r\n")
        )
        self.pending.clear()
        self.dropping = False
        return value

    def finish(self):
        return [self._take()] if self.pending or self.dropping else []


async def output_lines(stream):
    framing = BoundedLines()
    while chunk := await stream.read(8192):
        for line in framing.feed(chunk):
            yield line
    for line in framing.finish():
        yield line
