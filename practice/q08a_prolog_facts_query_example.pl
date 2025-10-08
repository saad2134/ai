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
% ?- likes(alice, pizza).
% ?- likes(X, pizza).
% ?- likes(alice, Food), food_type(Food, italian).
% ?- likes(X, Y), likes(Z, Y), X \= Z.