import requests
from lxml import etree
import csv
from selenium import webdriver
import pandas as pd
import os


class P2PEyeCrawler:
    def __init__(self):
        self._headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, '
                                       'like Gecko) fChrome/79.0.3945.130 Safari/537.36'}

    def create_csv(self, dataframe, name):
        dataframe.to_csv(os.path.join('/Users/holly/Desktop/毕设/Data/PlatformsComments', name + '.csv'), index=False)
        return 0

    def platformsBasicInfo(self, url):
        """url = 'https://www.p2peye.com/shuju/ptsj/'"""
        web = requests.get(url, headers=self._headers).text
        tree = etree.HTML(web)
        platform_table = tree.xpath('//table[@id="platdata"]')[0].xpath('//tr[@class="bd"]')

        platform_list = []
        count = 1
        for platform in platform_table:
            platform_dict = {}
            platform_dict['name'] = platform.find('.//a[@target="_blank"]').text
            platform_dict['reference'] = platform.find('.//a[@target="_blank"]').get('href')
            print(platform_dict['name'])
            print(count)
            count += 1
            platform_list.append(platform_dict)

        print(len(platform_list))
        df_platform = pd.DataFrame.from_dict(platform_list)
        return df_platform

    def singlePltComment(self, url):  # url截止到comment/
        comment_web = requests.get(url, headers=self._headers).text
        comment_tree = etree.HTML(comment_web)
        temp = comment_tree.xpath('//div[@class="c-page"]')[0].xpath('.//a')
        if len(temp) > 0:
            max_page = int(temp[-2].text)
        else:
            max_page = 1

        platform_all_comments = []
        count = 0
        for i in range(max_page):
            i_url = url + 'list-0-0-' + str(i+1) + '.html'
            i_web = requests.get(i_url, headers=self._headers).text
            i_tree = etree.HTML(i_web)
            comments_list = i_tree.xpath('//div[@class="floor"]')

            for comment in comments_list:
                comment_dict = {}
                comment_dict['user_name'] = comment.xpath('.//a[@class="qt-gl username"]')[0].text
                comment_dict['user_page'] = comment.xpath('.//a[@class="qt-gl username"]')[0].get('href')
                comment_dict['user_pid'] = comment_dict['user_page'].split('/')[-2][1:]
                if len(comment.xpath('.//div[@class="info clearfix"]')[0].xpath('.//div')) > 0:
                    comment_dict['major_tag'] = comment.xpath('.//div[@class="info clearfix"]')[0].xpath('.//div')[0].text
                tags_list = comment.xpath('.//li[@class="qt-gl"]')
                if len(tags_list) > 0:
                    tags = []
                    for j in range(len(tags_list)):
                        tags.append(comment.xpath('.//li[@class="qt-gl"]')[j].text)
                        tags.append(', ')
                    tags = ''.join(tags[:-1])
                    comment_dict['minor_tags'] = tags

                comment_dict['text'] = comment.xpath('.//a[@target="_blank"]')[1].text
                comment_dict['time'] = comment.xpath('.//div[@class="qt-gl time"]')[0].text

                comment_dict['num_like'] = comment.xpath('.//i')[0].text
                comment_dict['num_comments'] = comment.xpath('.//i')[1].text
                count += 1

                platform_all_comments.append(comment_dict)
            if count % 100 == 0:
                print("Comment number & Page:", count, i)
        df_single_platforms = pd.DataFrame.from_dict(platform_all_comments)

        return df_single_platforms

    def allPltComment(self):
        df_info = pd.read_csv('/Users/holly/Desktop/毕设/Data/PtfmNameURL.csv')
        df_info = df_info[68:]
        for index, row in df_info.iterrows():

                name = row['name']
                url = 'https:' + row['reference'][:-6] + 'comment/'
                df_cmt = self.singlePltComment(url)
                self.create_csv(df_cmt, name)
                print(name, "Finished.")

    def single_test(self, i_url):
        i_web = requests.get(i_url, headers=self._headers).text
        i_tree = etree.HTML(i_web)
        comments_list = i_tree.xpath('//div[@class="floor"]')
        count = 0
        for comment in comments_list:
            count += 1

            comment_dict = {}
            comment_dict['user_name'] = comment.xpath('.//a[@class="qt-gl username"]')[0].text
            comment_dict['user_page'] = comment.xpath('.//a[@class="qt-gl username"]')[0].get('href')
            comment_dict['user_pid'] = comment_dict['user_page'].split('/')[-2][1:]
            if len(comment.xpath('.//div[@class="info clearfix"]')[0].xpath('.//div')) > 0:
                comment_dict['major_tag'] = comment.xpath('.//div[@class="info clearfix"]')[0].xpath('.//div')[0].text
            tags_list = comment.xpath('.//li[@class="qt-gl"]')
            if len(tags_list) > 0:
                tags = []
                for j in range(len(tags_list)):
                    tags.append(comment.xpath('.//li[@class="qt-gl"]')[j].text)
                    tags.append(', ')
                tags = ''.join(tags[:-1])
                comment_dict['minor_tags'] = tags

            comment_dict['text'] = comment.xpath('.//a[@target="_blank"]')[1].text
            comment_dict['time'] = comment.xpath('.//div[@class="qt-gl time"]')[0].text

            comment_dict['num_like'] = comment.xpath('.//i')[0].text
            comment_dict['num_comments'] = comment.xpath('.//i')[1].text

            print(count)


def init_crawler():
    p2p_crawler = P2PEyeCrawler()
    """df, name = p2p_crawler.singlePltComment('陆金所')
    p2p_crawler.create_csv(df, name)"""
    p2p_crawler.allPltComment()
    return 0


if __name__ == '__main__':
    init_crawler()

