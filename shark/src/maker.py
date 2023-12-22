import pandas as pd

csv_test = pd.read_csv('/root/catkin_ws/src/shark/src/shark_map_final.csv')
print(csv_test)
df = pd.DataFrame(csv_test, columns = ['x', 'y', 'road', 'event'])

list_x = df['x'].values
list_y = df['y'].values
df['road'] = 0
df['event'] = 'run'

for i in range(len(list_x)-1):
    if(abs(list_x[i] - list_x[i+1]) > 0.1):
        if(abs(list_y[i] - list_y[i+1]) > 0.1):
            df.loc[i,'road'] = 'curve'
        else:
            df.loc[i,'road'] = 'straight'
    elif(abs(list_y[i]-list_y[i+1]) > 0.1):
        df.loc[i,'road'] = 'curve'
    else:
        df.loc[i,'road'] = 'stop'

for i in range(len(list_x)-1):
    if ((2850<=i<=2874) | (5025<=i<=5042) | (8545<=i<=8562)):
        df.loc[i, 'event'] = 'TrafficLight'

    if ((9931<=i<=10072)):
        df.loc[i, 'event'] = 'accel>=20'

    if ((3472<=i<=4500)):
        df.loc[i, 'event'] = 's_line'

    if (0<=i<=240):
        df.loc[i, 'event'] = 'L'
            
    if (250<i<300):
        df.loc[i, 'event'] = 'light_off'

    if (i == 458):
        df.loc[i, 'event'] = 'stop_stage'

    if (10933<i<10943):
        df.loc[i, 'event'] = 'R'

    if (10943 == i):
        df.loc[i, 'event'] = 'end_stage'

df.to_csv("/home/retta/catkin_ws/src/kimdom/rep/mando_map_final_marker.csv", index = False)
print(df)