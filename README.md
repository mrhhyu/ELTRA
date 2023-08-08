# CIKM23
This anonymous repository provides a proof for CRW scores' properties via the embedded .pdf file.

In this Appendix, we prove that the CRW scores are asymmetric, bounded, monotonic, unique, and always existent.

\noindent (1) \textbf{Asymmetry}: for every node-pair $(u,v)$ where $u\!\neq \!v$, $S(u,v)\! \neq \!S(v,u)$. \\
\textbf{\textit{Proof}:} According to Equation (2), if $u\!\neq\!v$, $S_k(u,v)$ is computed by considering $O_u$ and $I_u$, while $S_k(v,u)$ is computed by considering $O_v$ and $I_v$; since $O_u \!\neq\! O_v$ and $I_u \!\neq\! I_v$, then $S(u,v) \!\neq\! S(v,u)$ 

\noindent (2) \textbf{Bounding}: for all $k$,  $0 \!\le \!S_k(u,v) \!\le\! 1$. \\
\textbf{\textit{Proof}:} According to Equation (2), if $u\! \neq \!v$, then $S_0(u,v)\!=\!0$, otherwise $S_0(u,v)\!=\!1$; therefore $0 \!\le \!S_0(u,v)\! \le \!1$. It means the property holds for $k\!=\!0$. Now, we assume that the property holds for $k$, which means  $0\! \le\! S_k(u,v)\! \le\! 1$ for \textit{any} node-pairs $(u,v)$; according to the assumption $S_k(u,v) \!\ge\! 0$, thus
$$\begin{align*}
S_{k+1}(u,v) \! &=  \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} S_{k}(i,v)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}S_{k}(j,v)}{|I_u|}  \big) \\
& \ge  \! \frac{C}{2}\! \cdot\! \big(\frac{\sum_{i \in O_u} (0)}{|O_u|}\! +\! \frac{\sum_{j \in I_u}(0)}{|I_u|}  \big) \\
& \ge \! 0
\end{align*}$$
