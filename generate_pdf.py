"""
FLO OCR Sunum PDF Generator
Turkce karakter destekli - Hizalama duzeltilmis
"""

from fpdf import FPDF
import re
import os
import urllib.request
import zipfile

class SlidePDF(FPDF):
    def __init__(self):
        super().__init__(orientation='L', unit='mm', format='A4')
        self.set_auto_page_break(auto=False)
        self.set_margins(10, 10, 10)  # Daha dar marginler
        self.slide_num = 0
        self.total_slides = 0

        # Font dizini
        font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        os.makedirs(font_dir, exist_ok=True)

        dejavu_regular = os.path.join(font_dir, 'DejaVuSans.ttf')
        dejavu_bold = os.path.join(font_dir, 'DejaVuSans-Bold.ttf')
        dejavu_mono = os.path.join(font_dir, 'DejaVuSansMono.ttf')

        if not os.path.exists(dejavu_regular):
            print("DejaVu fontlari indiriliyor...")
            self._download_dejavu_fonts(font_dir)

        self.add_font('DejaVu', '', dejavu_regular)
        self.add_font('DejaVu', 'B', dejavu_bold)
        self.add_font('DejaVuMono', '', dejavu_mono)

        self.main_font = 'DejaVu'
        self.mono_font = 'DejaVuMono'
        self.content_width = 277  # A4 landscape - margins

    def _download_dejavu_fonts(self, font_dir):
        base_url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/"
        zip_name = "dejavu-fonts-ttf-2.37.zip"
        zip_path = os.path.join(font_dir, zip_name)

        try:
            urllib.request.urlretrieve(base_url + zip_name, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    if member.endswith('.ttf') and 'DejaVuSans' in member:
                        filename = os.path.basename(member)
                        target = os.path.join(font_dir, filename)
                        with zf.open(member) as src, open(target, 'wb') as dst:
                            dst.write(src.read())
            os.remove(zip_path)
            print("Fontlar yuklendi!")
        except Exception as e:
            print(f"Font indirme hatasi: {e}")
            self._use_windows_fonts(font_dir)

    def _use_windows_fonts(self, font_dir):
        import shutil
        win_fonts = "C:/Windows/Fonts"
        mappings = [
            ("arial.ttf", "DejaVuSans.ttf"),
            ("arialbd.ttf", "DejaVuSans-Bold.ttf"),
            ("consola.ttf", "DejaVuSansMono.ttf"),
        ]
        for src, dst in mappings:
            src_path = os.path.join(win_fonts, src)
            dst_path = os.path.join(font_dir, dst)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

    def footer(self):
        if self.slide_num > 0:
            self.set_y(-10)
            self.set_font(self.main_font, '', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 8, f'{self.slide_num} / {self.total_slides}', align='R')

    def _clean(self, text):
        text = text.replace('**', '').replace('`', '').replace('*', '')
        text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
        text = text.replace('⭐', '*').replace('✅', '[OK]').replace('❌', '[X]')
        text = text.replace('✓', '[OK]').replace('∉', '!=')
        return text.strip()

    def add_slide(self, title="", content="", is_title_slide=False):
        self.add_page()
        self.slide_num += 1
        title = self._clean(title)

        if is_title_slide:
            # Kapak slaytı
            self.set_fill_color(26, 95, 122)
            self.rect(0, 0, 297, 210, 'F')
            self.set_xy(10, 65)
            self.set_font(self.main_font, 'B', 32)
            self.set_text_color(255, 255, 255)
            self.multi_cell(self.content_width, 14, title, align='C')
            self.ln(5)
            self.set_font(self.main_font, '', 14)
            for line in content.split('\n')[:5]:
                line = self._clean(line)
                if line and not line.startswith('#'):
                    self.set_x(10)
                    self.cell(self.content_width, 8, line[:70], align='C')
                    self.ln()
        else:
            # Normal slayt
            self.set_fill_color(26, 95, 122)
            self.rect(0, 0, 297, 8, 'F')

            self.set_xy(10, 14)
            self.set_font(self.main_font, 'B', 22)
            self.set_text_color(26, 95, 122)
            self.cell(self.content_width, 10, title[:55], align='L')
            self.ln(14)
            self.set_x(10)
            self.render_content(content)

    def render_content(self, content):
        lines = content.split('\n')
        in_table = False
        table_data = []
        in_code = False
        code_block = []

        for line in lines:
            if self.get_y() > 185:
                break

            # Code block
            if line.strip().startswith('```'):
                if in_code:
                    self.render_code_block(code_block)
                    code_block = []
                in_code = not in_code
                continue

            if in_code:
                code_block.append(line)
                continue

            # Table
            if '|' in line:
                if '---' in line or all(c in '-|: ' for c in line):
                    continue
                cells = [c.strip() for c in line.split('|') if c.strip()]
                if cells:
                    table_data.append(cells)
                    in_table = True
                continue
            elif in_table:
                self.render_table(table_data)
                table_data = []
                in_table = False

            # ASCII diagram
            if any(x in line for x in ['└', '├', '┌', '─', '▶', '▼', '│', '-->']):
                self.set_font(self.mono_font, '', 8)
                self.set_text_color(50, 50, 50)
                clean = line
                for old, new in [('▶', '>'), ('▼', 'v'), ('─', '-'), ('└', '`'),
                                  ('├', '|'), ('┌', '.'), ('│', '|'), ('┐', '.'), ('┘', "'")]:
                    clean = clean.replace(old, new)
                self.set_x(10)
                self.cell(self.content_width, 4, clean[:80], align='L')
                self.ln()
                continue

            # H2
            if line.startswith('## '):
                self.ln(3)
                self.set_font(self.main_font, 'B', 14)
                self.set_text_color(44, 62, 80)
                text = self._clean(line[3:])
                self.set_x(10)
                self.cell(self.content_width, 8, text[:65], align='L')
                self.ln()
                continue

            # H3
            if line.startswith('### '):
                self.ln(2)
                self.set_font(self.main_font, 'B', 12)
                self.set_text_color(70, 70, 70)
                text = self._clean(line[4:])
                self.set_x(10)
                self.cell(self.content_width, 7, text[:65], align='L')
                self.ln()
                continue

            # Blockquote
            if line.startswith('>'):
                self.set_font(self.main_font, '', 11)
                self.set_text_color(80, 80, 80)
                text = self._clean(line[1:])
                self.set_x(15)
                self.multi_cell(self.content_width - 10, 6, text[:90], align='L')
                continue

            # Bullet points
            if line.strip().startswith('- ') or line.strip().startswith('* '):
                self.set_font(self.main_font, '', 11)
                self.set_text_color(44, 62, 80)
                text = self._clean(line.strip()[2:])
                self.set_x(12)
                self.cell(5, 6, '-', align='L')
                self.multi_cell(self.content_width - 15, 6, text[:85], align='L')
                continue

            # Numbered list
            match = re.match(r'^(\d+)\.\s*(.+)', line.strip())
            if match:
                self.set_font(self.main_font, '', 11)
                self.set_text_color(44, 62, 80)
                num, text = match.groups()
                text = self._clean(text)
                self.set_x(12)
                self.cell(8, 6, f"{num}.", align='L')
                self.multi_cell(self.content_width - 18, 6, text[:85], align='L')
                continue

            # Checkbox
            if '[ ]' in line or '[x]' in line:
                self.set_font(self.main_font, '', 11)
                text = line.replace('- [ ]', '[ ]').replace('- [x]', '[x]')
                text = self._clean(text)
                self.set_x(12)
                self.cell(self.content_width - 5, 6, text[:75], align='L')
                self.ln()
                continue

            # Normal text
            stripped = line.strip()
            if stripped and not stripped.startswith('<') and 'div' not in stripped:
                self.set_font(self.main_font, '', 11)
                self.set_text_color(44, 62, 80)
                text = self._clean(stripped)
                if len(text) > 2:
                    self.set_x(10)
                    self.multi_cell(self.content_width, 6, text[:100], align='L')

        if table_data:
            self.render_table(table_data)

    def render_table(self, table_data):
        if not table_data:
            return

        self.ln(3)
        num_cols = max(len(row) for row in table_data)
        max_width = self.content_width - 5
        col_width = min(max_width / num_cols, 50)

        # Tabloyu ortala
        table_width = col_width * num_cols
        start_x = 10 + (max_width - table_width) / 2

        for i, row in enumerate(table_data):
            self.set_x(start_x)
            if i == 0:
                self.set_font(self.main_font, 'B', 9)
                self.set_fill_color(26, 95, 122)
                self.set_text_color(255, 255, 255)
            else:
                self.set_font(self.main_font, '', 9)
                self.set_text_color(44, 62, 80)
                self.set_fill_color(250, 250, 250) if i % 2 == 0 else self.set_fill_color(255, 255, 255)

            for j, cell in enumerate(row):
                cell_text = self._clean(cell)[:20]
                self.cell(col_width, 7, cell_text, border=1, align='C', fill=True)
            self.ln()

        self.ln(3)

    def render_code_block(self, code_lines):
        if not code_lines:
            return

        self.ln(2)
        self.set_fill_color(40, 44, 52)
        self.set_text_color(180, 180, 180)
        self.set_font(self.mono_font, '', 8)

        y_start = self.get_y()
        max_lines = min(len(code_lines), 8)
        height = max_lines * 5 + 6

        self.rect(10, y_start, self.content_width, height, 'F')

        for i, line in enumerate(code_lines[:max_lines]):
            self.set_xy(14, y_start + 3 + i * 5)
            clean = line.replace('\t', '  ')[:70]
            self.cell(self.content_width - 10, 5, clean, align='L')

        self.set_y(y_start + height + 3)


def parse_markdown(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(r'^---\n.*?---\n', '', content, flags=re.DOTALL)
    slides = content.split('\n---\n')

    parsed = []
    for slide in slides:
        slide = slide.strip()
        if not slide:
            continue

        lines = slide.split('\n')
        title = ""
        body_lines = []

        for line in lines:
            if line.startswith('# ') and not title:
                title = line[2:].replace('**', '')
            else:
                body_lines.append(line)

        body = '\n'.join(body_lines).strip()
        parsed.append((title, body))

    return parsed


def main():
    print("FLO OCR Sunum PDF Generator")
    print("=" * 40)

    slides = parse_markdown('FLO_OCR_Sunum.md')
    print(f"Toplam {len(slides)} slayt")

    pdf = SlidePDF()
    pdf.total_slides = len(slides)

    for i, (title, body) in enumerate(slides):
        is_title = (i == 0) or ("tesekkur" in title.lower()) or ("sorular" in title.lower())
        pdf.add_slide(title, body, is_title_slide=is_title)
        print(f"  [{i+1:02d}] {title[:45]}")

    output_path = 'FLO_OCR_Sunum.pdf'
    pdf.output(output_path)
    print("=" * 40)
    print(f"PDF: {output_path}")


if __name__ == '__main__':
    main()
