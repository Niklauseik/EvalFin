import openai
import pandas as pd
import os
from utils import config_manager

# **读取 OpenAI API Key**
config_manager = config_manager.ConfigManager()
api_key = config_manager.get_api_key("openai")

client = openai.OpenAI(api_key=api_key)

# **构造 Prompt**
query = """"In the context of this series of interconnected finance-related queries and the additional information provided by the pretext, table data, and posttext from a company's financial filings, please provide a response to the final question. This may require extracting information from the context and performing mathematical calculations. Please take into account the information provided in the preceding questions and their answers when formulating your response:
Context: part ii item 5 2014market for registrant 2019s common equity and related stockholder matters market information . the common stock of the company is currently traded on the new york stock exchange ( nyse ) under the symbol 2018 2018aes . 2019 2019 the following tables set forth the high and low sale prices for the common stock as reported by the nyse for the periods indicated . price range of common stock . <table class='wikitable'><tr><td>1</td><td>2002 first quarter</td><td>high $ 17.84</td><td>low $ 4.11</td><td>2001 first quarter</td><td>high $ 60.15</td><td>low $ 41.30</td></tr><tr><td>2</td><td>second quarter</td><td>9.17</td><td>3.55</td><td>second quarter</td><td>52.25</td><td>39.95</td></tr><tr><td>3</td><td>third quarter</td><td>4.61</td><td>1.56</td><td>third quarter</td><td>44.50</td><td>12.00</td></tr><tr><td>4</td><td>fourth quarter</td><td>3.57</td><td>0.95</td><td>fourth quarter</td><td>17.80</td><td>11.60</td></tr></table> holders . as of march 3 , 2003 , there were 9663 record holders of the company 2019s common stock , par value $ 0.01 per share . dividends . under the terms of the company 2019s senior secured credit facilities entered into with a commercial bank syndicate , the company is not allowed to pay cash dividends . in addition , the company is precluded from paying cash dividends on its common stock under the terms of a guaranty to the utility customer in connection with the aes thames project in the event certain net worth and liquidity tests of the company are not met . the ability of the company 2019s project subsidiaries to declare and pay cash dividends to the company is subject to certain limitations in the project loans , governmental provisions and other agreements entered into by such project subsidiaries . securities authorized for issuance under equity compensation plans . see the information contained under the caption 2018 2018securities authorized for issuance under equity compensation plans 2019 2019 of the proxy statement for the annual meeting of stockholders of the registrant to be held on may 1 , 2003 , which information is incorporated herein by reference. .
Conversations: 
q0: what was the variance in the price of shares in the first quarter of 2002? answer: 13.73
Question: and what was it in that same period in 2001?"""

cot_prompt = (
    f"{query}\n\n"
    #"use Parkinson formula to calculate"
    "and asnwer : we are sure the answer 13.73 is right, what formual was used in the calculation of q0"
)

# **发送 API 请求**
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": cot_prompt}]
)

# **解析完整的 Response**
predicted_response = response.choices[0].message.content.strip()

# **打印结果**
print("🔍 CoT Analysis Result:\n")
print(predicted_response)


''' 
🔍 CoT Analysis Result:

To solve the final question step by step, we will proceed according to the steps outlined.

### Step 1: Identify the relevant financial figures from the provided data.
- From the data, we need to extract the high and low sale prices of the company's common stock during the first quarter of both 2001 and 2002. 
- The relevant figures for the first quarter of each year are:
  - **2001**: High = $60.15, Low = $41.30
  - **2002**: High = $17.84, Low = $4.11 
- These numbers are relevant as we are tasked with calculating the variance in the price of shares, which involves understanding the range of stock prices (i.e., the difference between the high and low prices) during those periods.

### Step 2: Determine the appropriate formula.
- The variance can be calculated by finding the price range, expressed as:
  \[
  \text{Price Range} = \text{High Price} - \text{Low Price}
  \]
- We will apply this formula to the respective years to find the price variance in the first quarter of 2001.

### Step 3: Compute the result.
- **For 2001**:
  \[
  \text{Price Range}_{2001} = \text{High}_{2001} - \text{Low}_{2001} = 60.15 - 41.30 = 18.85
  \]
- So, the variance in the price of shares in the first quarter of 2001 is **$18.85**.

### Step 4: Explain the role of the conversation history.
- The information provided in the previous question mentions that the variance in the price of shares in the first quarter of 2002 was $13.73. This insight provides a direct context about share performance over the years and allows for comparative analysis. 
- The reasoning that allows us to extend from the information in the previous question to our current question involves recognizing that we are using the same calculation method as before (i.e., determining the price range of share prices) but applying it to a different year. By applying consistent logic across these questions, we can derive their respective variances to understand stock performance more clearly.
'''