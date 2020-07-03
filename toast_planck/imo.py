# Copyright (c) 2015-2018 by the parties listed in the AUTHORS file.
# All rights reserved.  Use of this source code is governed by
# a BSD-style license that can be found in the LICENSE file.

import xml.etree.ElementTree as ET


class IMO():

    def __init__(self, imofile):
        self.imofile = imofile
        self.imo = ET.parse(imofile)

    def get(self, imopath, dtype=None):
        xpath = self._convert_to_xpath(imopath)
        elem = self.imo.find(xpath)
        value = elem.get('Value')

        if dtype is None:
            return value
        else:
            try:
                return dtype(value)
            except Exception as e:
                raise Exception('IMO: Failed to convert IMO value, {}, '
                                'to {}: {}'.format(value, dtype, e))

    def put(self, imopath, value):
        xpath = self._convert_to_xpath(imopath)
        elem = self.imo.find(xpath)
        try:
            for i, v in enumerate(value):
                elem[i] = v
        except Exception:
            elem = value

    def _convert_to_xpath(self, imopath):
        path_elem = imopath.split(':')
        new_path_elem = []
        for elem in path_elem:
            if ' ' in elem:
                new_elem = '[@'.join(elem.split(None, 1)) + ']'
            else:
                new_elem = elem
            new_path_elem.append(new_elem)
        xpath = '/'.join(new_path_elem)
        if xpath.startswith('IMO'):
            xpath = xpath.replace('IMO', '.')

        return xpath

    def save(self, imofile=None):
        if imofile is None:
            self.imo.write(self.imofile)
        else:
            self.imo.write(imofile)
