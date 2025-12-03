# 配置导出
# 1. 完整Conda环境
conda env export > environment.yml

# 2. 精简版pip依赖
pip list --format=freeze | findstr /V "conda" > requirements.txt

# 安装环境
conda create -n deep_ml python=3.9.25 -c conda-forge
conda activate deep_ml
python -m pip install -r requirements.txt