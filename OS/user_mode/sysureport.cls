\ProvidesClass{sysureport}
\LoadClass{ctexart}
\RequirePackage{xeCJK}
\RequirePackage{CJKnumb}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{titlesec}
\RequirePackage{fancyhdr}
\RequirePackage{listings}
\RequirePackage{xcolor}
\RequirePackage{array}
\RequirePackage{mathtools}
\RequirePackage{float}
\RequirePackage{indentfirst}
\RequirePackage{zhnumber}
\RequirePackage{booktabs}
\RequirePackage{enumitem}
\usepackage[linesnumbered,ruled,vlined,boxed]{algorithm2e}
\usepackage{minted}
\usepackage{subfigure}
\linespread{1.2}
\geometry{top=1in,bottom=1in,left=1in,right=1in}
\setcounter{secnumdepth}{3}
\renewcommand\thesection{\CJKnumber{\arabic{section}}、}
\renewcommand\thesubsection{\arabic{subsection}.}
\renewcommand\thesubsubsection{(\arabic{subsubsection})}


\setCJKmainfont[
  Path = ../../fonts/ ,
  Extension = .ttf ,
  BoldFont = SimHei ,
  ItalicFont = simkai ,
]{SimSun}

\titleformat*{\section}{\large\bfseries}
\titleformat*{\subsection}{\normalsize\bfseries}
\titleformat*{\subsubsection}{\normalsize}
\graphicspath{{figures/}{figure/}{pictures/}%
{picture/}{pic/}{pics/}{image/}{images/}}

\titlespacing{\subsection}{1em}{*4}{*1.5}
\titlespacing{\subsubsection}{1em}{*4}{*1.5}

% 代码格式和颜色定义
\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{comment}{rgb}{0.56,0.64,0.68}
\lstset{
  frame=tb,
  aboveskip=3mm,
  belowskip=3mm,
  xleftmargin=2em,
  xrightmargin=2em,
  showstringspaces=false,
  columns=flexible,
  framerule=1pt,
  rulecolor=\color{gray!35},
  backgroundcolor=\color{gray!5},
  basicstyle={\small\ttfamily},
  numbers=none,
  numberstyle=\tiny\color{gray},
  keywordstyle=\color{blue},
  commentstyle=\color{comment},
  stringstyle=\color{dkgreen},
  breaklines=true,
  breakatwhitespace=true,
  tabsize=2,
}

% 信息
\newcommand\major[1]{\def\@major{#1}}
\newcommand\name[1]{\def\@name{#1}}
\newcommand\stuid[1]{\def\@stuid{#1}}
\renewcommand\date[1]{\def\@date{#1}}
\newcommand\lab[1]{\def\@lab{#1}}
\newcommand\course[1]{\def\@course{#1}}
\newcommand\expname[1]{\def\@expname{#1}}
\newcommand\exptype[1]{\def\@exptype{#1}}

% 页眉
\pagestyle{fancy}
\lhead{实验名称：\@expname}
\chead{姓名：\@name}
\rhead{学号：\@stuid}


% 封面
\newcommand{\makecover}{
  \begin{center}
    \begin{center}
      \includegraphics[width=0.8\linewidth]{cover}
    \end{center}
    \vspace{1cm}
    \bfseries\huge{\@title}\\
    \vspace{3cm}
    \begin{center}
      \Large
      \linespread{2}
      \centering
      \renewcommand\arraystretch{1.6}
      \begin{tabular}{p{3cm}p{6cm}}
        \bfseries{课程名称：} & \@course \\ \cline{2-2}
        \bfseries{实验名称：} & \@expname \\ \cline{2-2}
        \bfseries{专业名称：}     & \@major   \\ \cline{2-2}
        \bfseries{学生姓名：}     & \@name \\ \cline{2-2}
        \bfseries{学生学号：}     & \@stuid  \\ \cline{2-2}
        \bfseries{实验地点：}     & \@lab  \\ \cline{2-2}
        \bfseries{实验成绩：} &   \\ \cline{2-2}
        \bfseries{报告时间：} & \@date  \\ \cline{2-2}
      \end{tabular}
    \end{center}
  \end{center}
  \thispagestyle{empty}
  \newpage
}
