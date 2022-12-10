# 生成html和markdown

## markdown
使用以下命令可以将所有的.ipynb文件转换为markdown文件
```
jupyter nbconvert --to markdown .\*\*.ipynb
```
执行完上述命令后，会为每个ipynb生成对应的markdown


## 通过mkdocs生成 在线文档
需要安装mkdocs和主题gitbook，我个人比较喜欢这个主题。
mkdocs用到了一个插件，所以也需要一起安装
```
pip install mkdocs
pip install mkdocs-gitbook
pip install mkdocs-exclude
```
在script目录下直接执行
```
mkdocs build
```
使用script目录中的mkdocs.yml配置即可生成web站点,可以在本地进行查看


## markdown 生成pdf

TODO