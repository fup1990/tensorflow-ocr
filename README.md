# tensorflow-ocr
使用tensorflow构建神经网络，识别图片中的文字。<br/>

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
 <table>
  <tr>
   <td>序号</td>
   <td>类型</td>
   <td>说明</td>
   <td>尺寸</td>
  </tr>
  <tr>
   <td>1</td>
   <td>图片</td>
   <td></td>
   <td>80*80*1</td>
  </tr>
  <tr>
   <td>2</td>
   <td>卷积层</td>
   <td>3*3/1</td>
   <td>80*80*64</td>
  </tr>
  <tr>
   <td>3</td>
   <td>池化层</td>
   <td>2*2/2</td>
   <td>40*40*64</td>
  </tr>
  <tr>
   <td>4</td>
   <td>卷积层</td>
   <td>3*3/1</td>
   <td>40*40*64</td>
  </tr>
  <tr>
   <td>5</td>
   <td>池化层</td>
   <td>2*2/2</td>
   <td>20*20*64</td>
  </tr>
  <tr>
   <td>6</td>
   <td>卷积层</td>
   <td>3*3/1</td>
   <td>20*20*128</td>
  </tr>
  <tr>
   <td>7</td>
   <td>池化层</td>
   <td>2*2/1</td>
   <td>20*20*128</td>
  </tr>
  <tr>
   <td>8</td>
   <td>全连接层</td>
   <td></td>
   <td>1024</td>
  </tr>
  <tr>
   <td>9</td>
   <td>全连接层</td>
   <td></td>
   <td>248</td>
  </tr>
 </table>
