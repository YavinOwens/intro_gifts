import os
import subprocess
import tempfile
import shutil
import pytest

def test_render_quarto_template():
    from skills_gap.tools.render_quarto_template import render_quarto_template
    # Use a temp directory to avoid polluting the real doc folder
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../standards/skills_gap_template.qmd'))
        output_html = os.path.join(tmpdir, 'skills_gap_template.html')
        output_pdf = os.path.join(tmpdir, 'skills_gap_template.pdf')
        # Call the render function
        render_quarto_template(template_path, tmpdir)
        # Check that the files exist
        assert os.path.exists(os.path.join(tmpdir, 'skills_gap_template.html'))
        assert os.path.exists(os.path.join(tmpdir, 'skills_gap_template.pdf')) 