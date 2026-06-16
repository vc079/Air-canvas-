import os, sys, traceback
print('cwd=', os.getcwd())
print('entries=', os.listdir('.'))
print('config_exists=', os.path.exists('config'))
if os.path.exists('config'):
    print('config_entries=', os.listdir('config'))
print('sys.path[0:5]=', sys.path[:5])
try:
    import importlib.util
    print('find_spec config=', importlib.util.find_spec('config'))
    print('find_spec config.gestures=', importlib.util.find_spec('config.gestures'))
    import config.gestures
    print('imported', config.gestures.__file__)
except Exception as e:
    print('import error:', type(e).__name__)
    traceback.print_exc()
