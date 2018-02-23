# tensorflow-ocr
使用tensorflow构建神经网络，识别图片验证码中的文字。<br/>

# 版本说明
 - python:3.5
 - tensorflow:1.4.1
 
 # 模块介绍
 - gen_captcha.py:生成图片验证码
 - word_vec.py:词向量处理
 - config.py:配置信息
 - cnn_train.py:神经网络模型训练
 - cnn_test.py:验证测试
 
 # 命令
 - 训练模型
 > python3 cnn_train.py
 - 验证测试
 > python3 cnn_test.py
 
 # 神经网络模型
