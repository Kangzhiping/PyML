import sys,string,types
import mechanize

#使用Mechanize 模拟浏览器

#sys.stdout=open('look','w')
file= open('list',"rb");
for name in file:
    br = mechanize.Browser()        #br是模拟的浏览器
    br.open('http://movie.douban.com/') #打开豆瓣电影页面
    br.select_form(nr=0)        #选一个表
    br.form['search_text']=name.decode('utf-8') #输入要查询的电影的名字
    br.submit()                #提交
    result = br.response()        #返回结果
    linkss = [l for l in  br.links()] #把浏览器链接加入linkss列表中
    rr = br.follow_link(linkss[21])   #点击搜索结果的第一条  这个21是尝试出来的，因为上面还有注册等等链接
    #print rr.read()
    ttt='<span class="pl">类型:</span> '  #手动找标签，也可以返回的源文件，用beautifulsoup解析
    #print br.title()
    ss=rr.read().split('\n')
    for line in ss:
        if line.find(ttt)>0:
            print (line)
    br.close()