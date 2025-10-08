% Family tree facts
parent(john, mary).
parent(john, tom).
parent(mary, susan).
parent(tom, jim).

% Gender facts
male(john). male(tom). male(jim).
female(mary). female(susan).

% Relationship rules
mother(X, Y) :- parent(X, Y), female(X).
father(X, Y) :- parent(X, Y), male(X).
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
sibling(X, Y) :- parent(P, X), parent(P, Y), X \= Y.

% ?- father(X, mary).        % Who is Mary's father?
% ?- mother(X, susan).       % Who is Susan's mother?
% ?- parent(john, X).        % Who are John's children?
% ?- grandparent(john, X).   % Who are John's grandchildren?
% ?- sibling(mary, tom).     % Are Mary and Tom siblings?
% ?- sibling(X, Y).          % Find all sibling pairs