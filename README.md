# visual_to_language

## gitIgnore说明
以下文件因为size太大，故仅作为本地保存，不同步到github
1. Project/data/
data下的文件为caption_train_images_20170902，challengerAI的原始数据
2. ThirdLibarary/neuraltalk2/vis/imgs/
这个是neuraltalk2在最终eval之后生成的结果图
3. ThirdLibarary/neuraltalk2/model/VGG_ILSVRC_16_layers.caffemodel 
neuraltalk2在运行时需要加入VGG_ILSVRC_16_layers.caffemodel，大小有500M

## 代码说明
1. 数据处理的主逻辑位于Project/ 路径下
2. Project/shell 下面的脚本，如果要运行需要修改执行路径
