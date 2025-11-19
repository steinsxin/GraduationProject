# 1. 完整Conda环境
conda env export > environment.yml

# 2. 精简版pip依赖
pip list --format=freeze | findstr /V "conda" > requirements.txt