# _*_ coding: utf-8 _*_
# time: 2024/2/27 16:36
# file: Prompting_glm4.py
# author: xandra

from zhipuai import ZhipuAI
import re
import json
client = ZhipuAI(api_key="")  
def ASK(ques):
    '''
    send the question to model glm-4 and get the answer
    Args:
        ques: what to ask.

    Returns:
        ans: the answer

    '''
    response = client.chat.completions.create(
        model="glm-4",  
        messages=[
            {
                "role": "user",
                "content": ques}
           ],
        # top_p=0.7,
        # temperature=0.95,
        max_tokens=50,
    )
    message = response.choices[0].message
    tokens = response.usage
    print(" --question:",ques)
    print(" --message:\n", message)
    # # print(response)
    print(" --total tokens:",tokens)
    patten = re.compile(r"((?<=(content=))(.*))(?= role)")
    result = patten.search(str(message))
    ans = result.group()
    return ans
def process_ans(ans):
    # delete useless charaters
    patten2 = re.compile('\(Word count: .*\)')
    ans = patten2.sub('',ans)
    ans = ans[1:-1]#delete ' '
    ans = ans.strip() #delete \n
    return ans
def promptForCls(attr_dict,cls):
    '''
    prompts each cls of the clslit, use the prompt temple(ques) constructed by the clsname and the attribute of attr_dict
    Args:
        attr_dict: attribute dict, consist of attribute type and attribute name
        cls: class list

    Returns:
        all_dict: all prompts for cls list

    '''
    all_dict = {} # save all prompts
    # generating prompts of each class
    for action in cls:
        # is action behavior or just physical/ environmental
        not_behavior = True  # default
        behavior_type = ['Actor','Body']
        print(f"START getting prompts of {action}...")
        for type in attr_dict.keys():
            if type == "Anomaly Specific Attributes":
                for attr in attr_dict[type].keys():
                    attribute = attr_dict[type][attr]
                    #prompt template
                    ques = f"What are the primary characteristics of [{action}] in term of its [{attribute}], explain in one sentence within 50 words."
                    print('GETING ANS...')
                    print(f'CLASS: {action}; ATTRIBUTE: {type},{attr}')
                    ans = ASK(ques)
                    if 'Behavioral Anomaly' or 'Behavioral anomaly' in ans: #behavior type
                        not_behavior = False
                    print(f"ans:{ans}")
                    ans = process_ans(ans)
                    # save to all_dict
                    if action in all_dict.keys():
                        all_dict[action][attr] = ans
                    else:
                        all_dict[action] = {}
                        all_dict[action][attr] = ans
                    # save tmp
                    with open(f"tmp_save/{action}_{attr}.txt", 'w') as f:
                        f.write(all_dict[action][attr])


            else:   # the other three types :Scene, Actor, Body
                # if not_behavior and type in behavior_type:
                #     print('- dont get its behavior type')
                #     continue # enviromental class don't get the action details type
                # else:
                #     print(f"- {type}")

                for attr in attr_dict[type]:
                    attribute = attr
                    ques = f"What are the primary characteristics of [{action}] in term of its [{attribute}], explain in one sentence within 50 words."
                    print('GETING ANS...')
                    print(f'CLASS: {action}; ATTRIBUTE: {type},{attribute}')
                    ans = ASK(ques)
                    print(f"  ans:{ans}")
                    ans = process_ans(ans)
                    # save to all_dict
                    if action in all_dict.keys():
                        all_dict[action][attr] = ans
                    else:
                        all_dict[action] = {}
                        all_dict[action][attr] = ans
                    # save tmp
                    with open(f"tmp_save/{action}_{attr}.txt", 'w',encoding='utf-8') as f:
                        f.write(all_dict[action][attr])
        print(f"FINISH getting prompts of {action}.")
    return all_dict
if __name__ == '__main__':

    # get attributes
    with open("attribute.json",'r') as f:
         attr_dict = json.load(f)
    # get classes
    cls = list(open("class_ubnormal.txt"))
    cls = [i.strip() for i in cls]
    # get the result of prompt
    all_dict = promptForCls(attr_dict,cls)

 # # save as dict->json to file
    print(all_dict)
    with open("prompts_ubnormal_all.json",'w') as f:
        json.dump(all_dict,f)






