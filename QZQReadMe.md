# NPU下安装LLamaFactory
pip install -e ".[torch-npu,metrics]"
pip install deepspeed==0.15.4

# 如果resume checkpoint有问题，可以试下
pip install transformers==4.50.0

# 可视化
pip install streamlit


# Github Push
如果遇到443，不妨尝试git config --global --unset https.proxy
# pip install
换清华源 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple streamlit
