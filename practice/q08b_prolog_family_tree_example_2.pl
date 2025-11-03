% --- Facts ---
male(harivansh). male(amitabh). male(abhishek).
female(teji). female(jaya). female(aishwarya). female(aradhya). female(shweta).
parent(harivansh, amitabh). parent(teji, amitabh). parent(amitabh, abhishek). parent(jaya, abhishek).
parent(amitabh, shweta). parent(abhishek, aradhya). parent(aishwarya, aradhya).

% --- Rules ---
mother(M, C) :- female(M), parent(M, C).
father(F, C) :- male(F), parent(F, C).
grandmother(GM, C) :- female(GM), parent(GM, M), parent(M, C).
grandfather(GF, C) :- male(GF), parent(GF, M), parent(M, C).
son(S, P) :- male(S), parent(P, S).
daughter(D, P) :- female(D), parent(P, D).
brother(B, S) :- male(B), parent(P, B), parent(P, S), B \= S.
sister(S, B) :- female(S), parent(P, S), parent(P, B), S \= B.

% --- Example Queries ---
% ?- father(F, abhishek).                       % Who is Abhishek's father?
% ?- mother(M, aradhya).                        % Who is Aaradhya's mother?
% ?- grandfather(GF, aradhya).                  % Who is Aaradhya's grandfather?
% ?- grandmother(GM, aradhya).                  % Who is Aaradhya's grandmother?
% ?- brother(B, shweta).                        % Who is Shweta's brother?
% ?- sister(S, abhishek).                       % Who is Abhishek's sister?
% ?- son(S, amitabh).                           % Who are Amitabh's sons?
% ?- daughter(D, amitabh).                      % Who are Amitabh's daughters?