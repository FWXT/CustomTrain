# data
论文分为多scale，代表reference token数目，越大用的token越少。并且scale 2会包含scale 4的数据，学习逐渐过渡由短到长。所以先scale 4训练。
# 基础data用各个scale下的train.json, valid.json, test.json
# 需要针对各个技术方案对上述json的字段进行修改使得效果好+模型小