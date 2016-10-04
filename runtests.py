# coding: utf-8

import unittest
import os
import shutil

from yast import Yast


class YastTest(unittest.TestCase):
    def test_all(self):
        sample = Yast('sample')

        sample.train([
            ('stock','英国脱欧与德银危机施压美股收跌'),
            ('stock','港股缺资金难闯24000点 美大选困扰后市'),
            ('f1', '2016丝绸之路拉力赛收官 标致道达尔汽车组夺冠'),
            ('f1','保时捷超级杯霍根海姆站 中国车手张大胜再出击'),
            ('basketball','林书豪透露生涯两低谷：效力湖人勇士令人失望'),
            ('basketball','后场双星合砍27分10助 开拓者全队发挥战胜爵士')])

        assert sample.predict_single('队内对抗曝光湖人新阵容 阿联或任内线主力替补').predicted_y == 'basketball'
        # basketball
        assert sample.predict_single('再出悲剧！ 达喀尔拉力赛后勤车肇事致1死10伤').predicted_y == 'f1'

        # cleanup
        if os.path.exists(sample.name+'.model'):
            shutil.rmtree('__pycache__')
            shutil.rmtree(sample.name+'.model')
            os.remove(sample.name+'_train.svm')


if __name__ == 'main':
    unittest.main()
