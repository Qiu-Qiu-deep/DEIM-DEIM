LaTeX Paper Writing Guidelines  必须遵守：
1. 文本使用中文。
2. 一定不要使用【item】分条，一定不要使用【textbf】给文本加粗，这些都是不规范的。
3. 论文是正式投稿论文，格式一定要规范且严谨，不要写的跟个报告似的一条接一条的，要连贯，一定要少分条，少用引号。
4. 在你添加参考文献时，一定要有根据这非常重要，你可以说不知道，然后向我询问，但是不能编造。
5. 添加表格或图片时，使用\resizebox{0.47\textwidth}{!}{}，方便我以后手动调整占比。
6. 对于【占位符】，用蓝色标记出来，定义成一个宏。
7. 对于占位图片，要使用类似\fbox{}，先画出个框来，同时将 \includegraphics[width=\linewidth]{figures/}也写上，但是先注释掉，占位图上应该写上这个图的内容应该是什么，好指导我放图。它的模板如下：
\begin{figure*}[!t]
\centering
% \includegraphics[width=\linewidth]{figures/XXX.pdf}
\fbox{\parbox{0.95\linewidth}{\centering \vspace{1.5cm} 占位符：XXX }}}
\caption{XXX}
\label{fig:XXX}
\end{figure*}
8. 对于一些可能会更换的名称，如何创新的模块名，尽量使用宏，在用宏时一定要注意，和后边的中文之间空一个格，不然会有语法错误。