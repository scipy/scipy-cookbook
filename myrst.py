"""reStructuredText Exporter class"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

from __future__ import division, absolute_import, unicode_literals, print_function

import re

from traitlets import default
from traitlets.config import Config

from nbconvert.exporters.templateexporter import TemplateExporter

import m2r


def my_convert(source, from_format, to_format, extra_args=None):
    if from_format == 'markdown' and to_format == 'rst':
        # handle math separately
        identifier_id = [0]
        identifier_fmt = "MATH{:09d}EXPRESSION"
        math_replacements = {}

        def add_identifier(text):
            identifier_id[0] += 1
            identifier = identifier_fmt.format(identifier_id[0])
            math_replacements[identifier] = text
            return identifier

        lines = source.splitlines()
        new_lines = []
        math_stack = []
        math_buf = []

        while lines:
            line = lines.pop(0)

            if math_stack:
                m = re.match(r'^\s*(\\end{equation}|\\end{align}|\$\$)\s*$', line)
                if m:
                    math_stack.pop()
                if line.rstrip().endswith('\\'):
                    line = line.rstrip() + '\\'
                math_buf.append("    " + line)
                if not math_stack:
                    math_buf.append("")
                    new_lines.append(add_identifier("\n".join(math_buf)))
                    del math_buf[:]
            else:
                m = re.match(r'^\s*(\\begin{equation}|\\begin{align}|\$\$)\s*$', line)
                if m:
                    if not math_stack:
                        math_buf.extend(["", ".. math::", ""])
                    math_stack.append(m.group(1))
                    math_buf.append("    " + line)
                else:
                    def math_replacement(m):
                        repl = r':math:`{}`'.format(m.group(1))
                        return add_identifier(repl)

                    line = re.sub(r'\$([^`]*?)\$', math_replacement, line)

                    if '$' in line and lines:
                        lines[0] = line + " " + lines[0]
                    else:
                        new_lines.append(line)

        if math_buf:
            new_lines.append(add_identifier("\n".join(math_buf)))

        source = "\n".join(new_lines)

        # convert
        output = m2r.M2R()(source)

        # re-insert math
        for k, v in math_replacements.items():
            output = output.replace(k, v)

        # return
        return output
    else:
        raise RuntimeError("Cannot convert from {} to {}".format(from_format, to_format))


class RSTExporter(TemplateExporter):
    """
    Exports reStructuredText documents.
    """

    @default('file_extension')
    def _file_extension_default(self):
        return '.rst'

    @default('template_file')
    def _template_file_default(self):
        return 'rst'

    output_mimetype = 'text/restructuredtext'

    @property
    def default_config(self):
        c = Config({'ExtractOutputPreprocessor':{'enabled':True}})
        c.merge(super(RSTExporter,self).default_config)
        return c

    def default_filters(self):
        filters = dict(super(RSTExporter, self).default_filters())
        filters['convert_pandoc'] = my_convert
        return filters.items()
