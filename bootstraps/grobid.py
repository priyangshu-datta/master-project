from icecream import ic
import os
from pathlib import Path

def grobid_init(grobid_version):
    try:
        if not Path(f'grobid-{grobid_version}/').exists():
            if not Path(f'{grobid_version}.zip').exists():
                os.system(f'wget https://github.com/kermitt2/grobid/archive/{grobid_version}.zip')    
            os.system(f'unzip {grobid_version}.zip')
            os.remove(f'{grobid_version}.zip')
        if Path(f'grobid-{grobid_version}/grobid-core/build/libs/grobid-core-{grobid_version}-onejar.jar').exists():
            return True
        os.chdir(f'grobid-{grobid_version}/')
        os.system('./gradlew clean install')
        os.chdir('/workspaces/master-project')
        os.system('clear')
        return True
    except Exception as e:
        ic(e)
        return False