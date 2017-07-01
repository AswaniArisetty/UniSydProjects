
### Effective Data Visualization with D3 / Dimple.js

##### Data Set :

Titanic data from Udacity has been used for this analysis. I have previously done exploratory and statistical analysis using the same data and hence used the same analysis and summary to create the visualization.

##### Summary  :

From the EDA and statistical analysis (attached the relevant ipython notebook / html) 

1. Women had higher survival rate compared to men across all Cabins.
2. Cabin 3 had the least number of survivors with almost 74% deceased in the tragedy.

The visualization has been designed primarily to show the above two points.

##### Design Decisions :

As this is mostly Categorical data (Cabins and Genders) , I have decided to use a Bar chart initially , and wanted to show Survivors / Non Survivors in different colours . But have eventually decided to use a Stacked Bar chart. The graph by default shows the Passenger Survival counts across the Cabins. There are buttons for user to choose Male or Female survival counts and percentages individually. A legend describes the colours for Survived / Deceased. Also a tool tip appears when we hover over the chart.

#### Feedback :

I have taken the feedback from my family members and a few colleagues and have summarized the main diffrences between the initial and final charts as below. As this is a not so complex chart , people were able to understand the information but had feedback about look and feel.

1. A clear description and unintuitive axes labels were used. Using column names as axes labels did not help my friends who were not familiar with the data.
2. Buttons were not highlighted as well as the cursor did not change when hovered. No one knew that its interactive till I informed them.
3. Default tool tip was not proper while displaying "Deceased" Category.
4. Initially selected buttons need to be highlighted when the chart loads.(All and Count).
5. Layout needs to be changed.
6. Percentage information seems to give a better understanding of the Summary than the counts although it is good to have counts to get actual numbers.

I have incorporated all the feedback in the index_final.html. 

##### References :

1. Visual Storytelling with D3 by Ritchie King.
2. Udacity Data visualization course.
3. Lot of Mike Bostock tutorials.
4. Dimple.js documentation
5. Stack overflow
6. Few other books and snippets on Safari books.

