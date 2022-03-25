from secrets import token_urlsafe
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
import re
import torch.nn.functional as F

def pre_ac(text):
    ret = []
    texts = text.split('|||')
    for text in texts:
        url_pattern = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]'
        text = re.sub(url_pattern, '', text)
        # tag_pattern = '#[a-zA-Z0-9]*'
        # text = re.sub(tag_pattern, '', text)
        at_pattern = '@[a-zA-Z0-9]*'
        text = re.sub(at_pattern, '', text)
        not_ascii_pattern = '[^a-zA-Z0-9]'
        text = re.sub(not_ascii_pattern, ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()
        ret.append(text)
    return ret

target = 4 - 1

text = \
'''
pokinometry is as easy as 1-2-3 !|||just follow these simple steps : # 1 : decide if you want white rice , brown rice , salad , chips or any two for half and half # 2 : choose 2 scoops of fish for small , 3 scoops medium or 5 scoops for a large bowl # 3 : would you like some sauce ?|||if yes , spicy or non-spicy ?|||spicy mayo is new .|||get it !|||# 4 : how about some toppings ?|||smelt eggs , onions , and sesame seeds are the basic # 5 : pay up and enjoy your creation !|||fish choices include : tuna , salmon , yellow tail , albacore , shrimp , octopus , scallop , mashed spicy tuna ( all raw except for the shrimp and octopus ) i for one went with the large bowl with half white and half brown rice .|||slices of cucumbers , some avocado , onions , and imitation crab are automatic .|||do n't skip those .|||i got two scoops of salmon , a scoop of mashed spicy tuna , octopus , and yellow tail .|||everything was harmoniously delicious that i 'll pick exactly the same fish all over again in a heartbeat .|||i love a good kick for some excitement so i asked for medium spice level plus their spicy mayo sauce to be added .|||boy was my tongue on fire .|||love it !|||of course , all toppings added but with ginger on the side .|||i have to say , i outdid myself with my own genius creation .|||bravo !|||the concept of customizing your own poke bowl is utterly brilliant .|||it 's one of those things where you 'd ask yourself : `` now , why did n't i think of that ? ''|||my thoughts exactly !|||overhead expense is low with it being a self-service establishment and no hot kitchen needed .|||it 's affordable so the target market is pretty much everyone who enjoys eating raw fish .|||with disposable bowls and utensils , you can easily decide to eat in or have it to-go .|||the ambiance is clean and simplistic with no point for guests to linger and take their time , thus making the turnover pretty quick .|||only caveat is the parking .|||otherwise , this business is the shiznit !'
'''


text2 = \
    '''
    谈下使用f4一周的感受，是第二轮在天猫抢购到的，过了五六分钟还能抢到，说明备货还是很足的，不像某些大厂一样玩饥饿营销。说下主要有点：1、599的指纹机，不知道算不算最低的，不过绝对良心了；2、整体手机握感很舒服，虽然只有边框是金属的，但是还挺有质感额；3、360OS的系统点个赞，财产隔离、冷藏室之类的黑科技不错。。不足就是内存有点不够用，不过599能这配置也算良心了，要是早点出高配版就好了~·。599元的价格，有金属、有指纹、有双微信，还是4G，还要啥自行车呢。 
    '''

# text = text2
texts = pre_ac(text)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert = BertModel.from_pretrained(
    'bert-base-uncased', 
    config='configs/config_demo.json', 
    add_pooling_layer=True
)
cls_head = nn.Sequential(
            nn.Linear(
                bert.config.hidden_size, 
                bert.config.hidden_size
            ),
            nn.ReLU(),
            nn.Linear(
                bert.config.hidden_size, 
                5
            )
        )

text3 = \
    '''
    流畅度高。屏幕感觉还没有g2的感觉好。第一篇原创，如有不足请各位包涵，指正！.        最近一直计划给老婆大人换手机，关注了很久，由于老婆不喜欢用苹果，所以从上一代旗舰的s8和索大法看到最近的一加6和小米MIX2s。最后定下买首发的一加。.       下面简单上几张图片，供大家参考(由于本人对拍照毫无兴趣，照片质量不是很好，大家将就看吧！.内部配件总览.墨岩黑背面单独来一张.开机.手机桌面.先说一下收到后的外观感受吧，墨岩黑的玻璃后壳看上去很有质感，手感也不错。论坛上说边框些割手，从手机背面往前面划的时候，会有割手的感觉，主要是手机背壳和边框不是完全对上的，不知道是装配工艺的问题还是就这样。其他方面还算满意吧。.       然后是屏幕问题，不说和苹果比，我上一个安卓手机是LG G2，感觉还没有g2的感觉好。颗粒感还是比较严重的。话说我还是比较喜欢is屏的感觉。.       手机流畅性还是么有问题的，毕竟845+8G的组合，游戏的话我没有体验，以后有时间再写吧。.       电池在我下午三点收到货的时候剩余电量50左右，下了一些软件，看了半个小时的直播，其余大部分时间是待机，到第二天八点左右剩余30左右的电量。.        关于一加6的上手体验就写这些了，如果有还想了解的问题，可以留言，看到后我会尽量回复的！.
    '''
tokens = tokenizer(text3, padding='longest', return_tensors="np")  
print(tokens.input_ids)


# print(tokens.input_ids)
# print(split_words(tokens.input_ids))
# 101 [CLS] 102 [SEP]