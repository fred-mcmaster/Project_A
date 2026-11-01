"""Execute and save the project notebook, keeping genuine cell outputs and live logs."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for key, subfolder in [('MPLCONFIGDIR', 'matplotlib'), ('IPYTHONDIR', 'ipython'),
                       ('JUPYTER_RUNTIME_DIR', 'jupyter_runtime')]:
    cache = ROOT / '.work' / subfolder
    cache.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(cache)
os.environ['JUPYTER_PATH'] = str(ROOT / '.venv' / 'share' / 'jupyter')
import nbformat
from nbclient import NotebookClient

path = ROOT / 'ECE1512_ProjectA_Student.ipynb'
nb = nbformat.read(path, as_version=4)


class LoggedClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)


def save_cell(cell, cell_index, **kwargs):
    nbformat.write(nb, path)
    print(f'[Saved cell {cell_index}]', flush=True)


if __name__ == '__main__':
    client = LoggedClient(nb, timeout=None, kernel_name='python3',
                          resources={'metadata': {'path': str(ROOT)}},
                          on_cell_executed=save_cell)
    try:
        client.execute()
    finally:
        nbformat.write(nb, path)
    errors = [output for cell in nb.cells if cell.cell_type == 'code'
              for output in cell.outputs if output.output_type == 'error']
    if errors:
        raise RuntimeError(f'{len(errors)} notebook errors')
    print('Notebook executed successfully and saved with outputs.', flush=True)
