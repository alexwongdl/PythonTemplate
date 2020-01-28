### 图像分割resize主要部分

#### 图像分割udf:
* image_algorithm_platform:Image_Segmentation_Ratio(tfs, "image_segment_related.yml","true","True") as seg

* bi_udf:bi_get_json_object(a.seg, '$.tfsfilename') as seg_tfs,
* bi_udf:bi_get_json_object(seg, '$.m_iState') as seg_state,
  seg_state=1表示成功

#### 目标
把图片主要部分缩放后放到800*800的白底图片中,主要部分上边界距离白底图上边界220, 距离下边界140, 缩放后过宽(暂定760)的图片丢弃,宽过小的图片(暂定300)丢弃.

#### 数据表
* tyx.market_material_all -- white_graph
* tbods.s_auction_extends where extends_key='white_bg_image'---value_start
 join tbcdm.dim_tb_itm 取叶子类目id=50019790
