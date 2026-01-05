# 1. 检查 JSON 是否合法
python -c "import json; print(json.load(open('data/gwhd_2021/annotations/test.json')))"

# 2. 检查 category_id 是否都是 0
python -c "
import json
with open('data/gwhd_2021/annotations/test.json') as f:
    anns = json.load(f)
    cats = [a['category_id'] for a in anns['annotations']]
    print('Category IDs:', set(cats))
"

# 3. 检查图像是否存在
python -c "
import os
import json
with open('data/gwhd_2021/annotations/test.json') as f:
    anns = json.load(f)
    img_paths = [os.path.join('data/gwhd_2021/test/images', img['file_name']) for img in anns['images']]
    missing = [p for p in img_paths if not os.path.exists(p)]
    if missing:
        print('Missing images:', missing)
    else:
        print('All images exist')
"