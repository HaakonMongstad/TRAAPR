## <strong> Game mechanics

# Actions and Action Space
0  - Empty 

1  - Wall 

2  - Friendly

3 - Enemy
    
# Rewards
Rewards/Punishments
 - Losing a cell to overpopulation (Surrounding) -1
 - Agent death (enemy take over) -2
 - All squares eliminated -100
 - Eliminating all enemy squares +100
 - Attacking
    - Enemy
        - Win +5
        - Tie -2
        - Loss -3
    - Neutral
        - Win +3
        - Tie -1
        - Loss -3
    - Team
        - Win +3
        - Tie +5
        - Loss -5
# State Transition Probability
Possibly determine the odds an agent can take a square based on how many other agents there are around. 
Ex:
<p> [2,2,2,2,2]
<p> [2,2,2,2,2]
<p> [2,2,2,0,2]
<p> [2,2,2,2,2]
<p> [2,2,2,2,2]
<p> 100% chance of taking over the square with a 0
