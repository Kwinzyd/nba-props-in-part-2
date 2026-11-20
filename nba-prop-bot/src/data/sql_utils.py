"""SQL schema statement parsing without optional database-driver imports."""

def split_schema_sql(sql):
    """Split our schema without cutting quoted PL/pgSQL bodies or strings."""
    import re
    statements, chunk, quote, dollar = [], [], None, None
    index, comment_depth = 0, 0
    while index < len(sql):
        if comment_depth:
            if sql.startswith('/*', index):
                comment_depth += 1
                index += 2
            elif sql.startswith('*/', index):
                comment_depth -= 1
                index += 2
            else:
                index += 1
            continue
        if dollar:
            if sql.startswith(dollar, index):
                chunk.append(dollar)
                index += len(dollar)
                dollar = None
            else:
                chunk.append(sql[index])
                index += 1
            continue
        ch = sql[index]
        if quote:
            chunk.append(ch)
            if ch == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote:
                    chunk.append(quote)
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if sql.startswith('--', index):
            newline = sql.find('\n', index)
            index = len(sql) if newline < 0 else newline
            continue
        if sql.startswith('/*', index):
            comment_depth = 1
            chunk.append(' ')
            index += 2
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == '$':
            match = re.match(r'\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$', sql[index:])
            if match:
                dollar = match.group()
                chunk.append(dollar)
                index += len(dollar)
                continue
        elif ch == ';':
            if ''.join(chunk).strip():
                statements.append(''.join(chunk).strip())
            chunk = []
            index += 1
            continue
        chunk.append(ch)
        index += 1
    if quote or dollar or comment_depth:
        raise ValueError('unterminated schema quote or comment')
    if ''.join(chunk).strip():
        statements.append(''.join(chunk).strip())
    return statements

