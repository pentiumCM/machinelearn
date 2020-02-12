#!/usr/bin/env python
# encoding: utf-8
'''
@Author  : pentiumCM
@Email   : 842679178@qq.com
@Software: PyCharm
@File    : TestDict.py
@Time    : 2019/12/17 14:35
@desc	 : 单元测试
'''

import unittest

from test.mydict import Dict


class TestDict(unittest.TestCase):

    def test_init(self):
        print("hello World1")
        d = Dict(a=1, b='test')
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 'test')
        self.assertTrue(isinstance(d, dict))

    def test_key(self):
        print("hello World2")
        d = Dict()
        d['key'] = 'value'
        self.assertEqual(d.key, 'value')

    def test_attr(self):
        print("hello World3")
        d = Dict()
        d.key = 'value'
        self.assertTrue('key' in d)
        self.assertEqual(d['key'], 'value')

    def test_keyerror(self):
        print("hello World4")
        d = Dict()
        with self.assertRaises(KeyError):
            value = d['empty']

    def test_attrerror(self):
        print("hello World5")
        d = Dict()
        with self.assertRaises(AttributeError):
            value = d.empty

    def test_hello(self):
        print("hello World")

    def setUp(self):
        print('setUp...')

    def tearDown(self):
        print('tearDown...')

    if __name__ == '__main__':
        unittest.main()
