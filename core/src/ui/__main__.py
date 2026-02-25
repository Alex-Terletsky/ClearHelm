import sys
import os

# When run as `python core/src/ui`, Python executes this file without package
# context, so relative imports fail. Add core/src to sys.path so the ui
# package and its siblings (params, runner, manager) are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.main_window import main

main()
