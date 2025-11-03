likes(alice, pizza).
likes(alice, sushi).
likes(bob, pasta).
likes(bob, pizza).
likes(carol, pizza).
likes(carol, salad).

food_type(pizza, italian).
food_type(pasta, italian).
food_type(sushi, japanese).
food_type(salad, healthy).

% Example queries:
% ?- likes(alice, pizza).                               % Does Alice like pizza?
% ?- likes(X, pizza).                                   % Who likes pizza?
% ?- likes(alice, Food), food_type(Food, italian).      % Which Italian foods does Alice like?
% ?- likes(X, Y), likes(Z, Y), X \= Z.                  % Which different people like the same food?
% ?- food_type(Food, italian), likes(bob, Food).        % What Italian foods does Bob like?
% ?- likes(Person, sushi), food_type(sushi, Type).      % Who likes sushi, and what type of food is it?
% ?- food_type(Food, healthy), likes(Person, Food).     % Who likes healthy food?
% ?- likes(Person, Food), food_type(Food, japanese).    % Who likes Japanese food?