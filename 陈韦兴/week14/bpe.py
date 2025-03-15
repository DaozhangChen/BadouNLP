text = """公园1872之夜暨招商地产25周年庆典酒会(组图)
    2009年4月25日19：00，地处朝阳区东四环红领巾公园湖畔的公园1872 (论坛 相册 户型 样板间 地图搜索)营销中心高朋满座，展开一幅“百年之上·传世珍品”公园1872璀璨之夜--暨招商地产(企业专区,旗下楼盘)25周年庆典答谢酒会的盛世画卷。    招商局地产北京管理总部营销总监李杰先生对二十五年来一直支持和关心招商地产的新老客户、各界朋友表示真挚的谢意并表示招商地产会秉承“百年招商家在情在”的企业理念更加努力和精工细作，提供让客户更加满意的人居空间。    酒会现场，上演了精彩纷呈的vertu手机和邦克珠宝奢华秀，给予客户欣赏到国际一线奢侈品牌的时尚视觉盛宴。红酒鉴赏、互动游戏、职业魔术表演等更给予客户眼花缭乱的视觉冲击力，愉悦的名酒品鉴之旅，生动的互动游戏让整个酒会现场华彩满堂。    国际奢侈品牌竞猜活动更让客户扣动心弦，热情澎湃，真实地体验到招商地产的厚重奢侈回馈，Prada、Gucci、Armani 、范思哲等名品回馈，让客户由衷的体验奢侈之旅，尤其是万元以上的PRADA紫色时尚新款手提包让客户兴奋不已，并在浪琴、欧米茄的国际名品竞猜中让客户笑颜收囊而归。　
    时间：
    2009年4月25日19：00——21：00
    地点：
    公园1871售楼处
    以下为本次活动的精彩实录：
图为活动开场节目“激情小提琴演奏”
　　主持人：尊敬的各位女士们、先先们，大家晚上好！首先非常欢迎各位来参加百年之上·传世珍品，公园1872璀璨之夜暨招商地产25周年庆典答谢酒会，主持人姚姚代表活动的主办方对各位的到来表示衷心的感谢！今天是个非常特别的日子，招商地产迎来自己的25周岁生日，可以说招商地产从1984年发展至今，在经过市场不断洗礼和考验之后，招商地产已经发展成为集开发、物业有机结合，物业品种齐全的房地产企业集团，形成以深圳为中心，珠三角、长三角和环渤海经济带为重点经营区域的市场格局。
　　25周年对于招商地产来讲也是非常短暂，从青年步入中年，同时也迎来自己的成熟和辉煌，在这个非常喜庆的日子里，我们首先掌声有请公园1872营销总监李杰为我们致辞。
　　李杰：谢谢主持人，今天是个特别的夜晚，非常荣幸能够邀请到我们尊贵的客户，还有华夏银行的各位贵宾来参加招商地产25周年的庆典酒会，25年前，就是1984年4月25日是招商地产成立的日子，25年对于很多房地产企业来说很长，但是对于招商地产来说时间并不长，其实在这之前招商地产已经走过了100多年的历程，所以招商地产现在虽然只有25年，但是流淌的是100多年的企业血液。
图为公园1872营销总监李杰
　　这100多年的发展经验以及它的公司文化，在招商地产中充分得到了体现。08年对招商地产是非常重要的一年，08年遇到了很多大的事情，在市场冲击的情况下能够得到各位客户的认可，这是招商地产最大的收获。09年对招商地产来说也是非常重要的一年，09年是招商地产的客户服务年。虽然国内的房地产公司客户服务体系从招商地产开始的，但是25年历程中我们觉得依然不够，从今年开始我们会把客户服务作为招商地产最重要的经营指标来去对待，而不是简单的是个工作任务，经过今年以后，我想招商地产的服务能够更上一个台阶，希望能够得到各位客户更大的认可。
　　招商地产经过这25年，我也希望下一个25年，再下一个25年，跟各位还能再见面，希望你们成为我们永远的客户。今天我就讲到这里，谢谢各位的参加。
　　主持人：谢谢李杰先生的致辞，让我们感受到25年当中招商地产所走过的一段不同寻常的历程，同时也与我们一同展望未来更远之后，我们招商地产即将取得的辉煌历程。其实在今天这个夜晚，我们邀请各位和我们一同，不但要见证招商地产所走过的25周年的辉煌，更重要的是要为在场的每一位朋友去打造一段属于你们最美好的璀璨之夜的记忆。接下来的时间就让我们跟随模特的脚步一同欣赏一段珠宝秀，相信这段珠宝秀必将点亮今晚璀璨的星空，有请！
我要评论

"""


def text_encode(text):
    ## 生成utf8编码
    return text.encode("utf8")

def generate_bpe(encode_list,token_map):
    ## 找出当前最多的pair
    max_pair = find_max_pair(encode_list)
    ## 记录最新的pair
  
    if max_pair:
        if token_map :
            max_number = token_map[max(token_map,key = lambda k : token_map[k])]
            token_map[max_pair] = max_number + 1
            update_number = max_number + 1
        else:
            token_map[max_pair] = 256
            update_number = 256
    
        return update_text_encode(encode_list,max_pair,update_number)
    
    else:
        return encode_list 


def find_max_pair(encode_list):
    encode_map = {}
    for pair in zip(encode_list,encode_list[1:]):
        encode_map[pair] = encode_map.get(pair,0) + 1

    ## 防止出现所有值都为1，仍然还继续进行压缩的情况
    if len(set(encode_map.values())) == 1:
        return None
    else:
        return max(encode_map,key=lambda k:encode_map[k])


def update_text_encode(text_encode,max_pair,pair_number):
    ## 更新原encode
    number_list = []
    current_index = 0
    while current_index < len(text_encode):
        if current_index == len(text_encode) - 1:
            number_list.append(text_encode[current_index])
            current_index += 1
        else :
            if text_encode[current_index] == max_pair[0] and text_encode[current_index + 1] == max_pair[1]:
                number_list.append(pair_number)
                current_index += 2
            else:
                number_list.append(text_encode[current_index])
                current_index += 1
    return number_list


def tokenizer(text,token_map):
    ## 给输入文本编码
    encode = list(text_encode(text))
    continue_token = True
    index = 0
    

    while continue_token:
        current_encode = []
        continue_token = False

        while index < len(encode):
            if index == len(encode) - 1:
                current_encode.append(encode[index])
                index += 1
            else:
                pair = (encode[index] , encode[index + 1])
                token_number = token_map.get(pair,None)
                if isinstance(token_number,int):
                    current_encode.append(token_number)
                    if not continue_token:
                        ## 如果还能够找到token，那就继续下一轮分词
                        continue_token = True
                    index += 2
                else:
                    current_encode.append(encode[index])
                    index += 1
                
        index = 0
        encode = current_encode    
    
    return encode





if __name__ == "__main__":
    ## 形成bpe
    limit_token_number = 1500 ## 限制token的总数量

    token_map = {}
    encode_list = list(text_encode(text))
    new_encode_list = []
    
    while len(token_map) <= limit_token_number:
        new_encode_list = generate_bpe(encode_list,token_map)
        if len(new_encode_list) == len(encode_list):
            break
        else:
            encode_list = new_encode_list
    
    print(token_map)

    ## 使用token_map进行编码
    token = tokenizer("公园1872之夜暨招商地产25周年庆典酒会(组图)",token_map)
    # print(token_map)
    print(token)
