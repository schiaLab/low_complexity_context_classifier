import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



freq = pd.read_csv("data/freq.csv") #같은 폴더 안에 있는 freq.csv 파일을 읽어 판다스 객체로 변환했습니다.


print(freq)#시범적으로 freq 객체를 인쇄해 보았습니다. 정상적으로 작동한다면, 다음으로 넘어가도 좋습니다.

class DecomposeLetter:
#클래스는 기본적인 자료형의 변형과 응용을 위해서도 필수적이지만, 우리가 직접 클래스를 정의하여 타이핑하는 코드의 양을 획기적으로 줄일 수 있습니다.
#이 클래스의 기능은 들어오는 자료형의 음절 개수 분석의 반복을 줄이기 위함입니다.

    #이 클래스가 하나의 문자열을 완전히 분해하여 음절 단위의 사용 빈도수로 변환한다는 사실을 기억하면서, 필요한 애트리뷰트를 선언합시다.
    origin : str #이 애트리뷰트는 객체가 분석할 원문, 즉 오리지날 데이터가 들어갈 자리입니다. 데이터는 문자열일 것이기에 문자열로 선언했습니다.
    data : dict #이 애트리뷰트는 객체가 분석하고 그 결과를 기록할 일종의 스프레드시트입니다. 앞서 보셨던 판다스 등 데이터과학에서 활용하는 다양한 자료형이 있지만,
                #여기선 일단 딕셔너리를 사용하도록 하겠습니다. 딕셔너리의 구성은 하나의 문자(key)에 대해서 그 빈도수(value)가 될 것입니다.
    deltaResult : list #이 애트리뷰트는 객체가 분석한 데이터와 일반인들의 음운 사용 패턴간의 차이를 내보이기 위한 데이터입니다. 지금은 리스트로 해두겠습니다.

    def __init__(self, origin):

        self.origin = origin
        self.data = dict()
        self.sortedData = None
        self.standardData =None

    def decompose(self): #원문을 분해할 메소드를 선언했습니다.


        for letter in self.origin: #원문의 자료형은 문자열, 즉 시퀀스이므로 for 루프를 돌 수 있습니다.
                                    #이 for 루프는 모든 영문자 하나하나를 돌 겁니다.

            self.data[letter.lower()] = self.data.get(letter.lower(), 0) + 1





                                    #이 코드의 기능은 self.data 딕셔너리에 찾은 영문자 하나가 있을 경우
                                                        # 그 개수를 하나추가하고, 그렇지 않을 경우 그 글자를 키로 하는 쌍을 하나 더 생성합니다.
                                                        # .lower()는 해당 문자열을 소문자로 만니다.
        for letter in "qwertyuiopasdfghjklzxcvbnm": #데이터 클랜징 과정으로, 이제 우리 객체의 데이터에 혹시나 빠졌을 글자 데이터에 0을 채워주는
            #작업을 해보겠습니다. 제가 알파벳 순서를 외울 줄 몰라 키보드 순서대로 모든 알파벳을 눌렀습니다. 문자열도 시퀀스이므로, for 문을 쓸 수 있습니다.

            self.data[letter] = self.data.get(letter, 0) #위와 유사해 보이지만 이번엔 이미 존재하는 글자에 대해선 숫자를 변경하지 않고,
                                                        #글자가 없을 경우 0을 대입합니다.





    def vector(self): #기존 빈도수와 객체의 데이터와의 빈도 차이를 delta에 기록하는 메소드를 선언합니다.
        sum = 0
        aList = list() #임시 리스트입니다.
        bList = list()
        for value in self.data.values(): #data 딕셔너리에서 그 빈도 값만 추출했습니다.

            sum += value #기존 데이터값이 빈도수 비율이므로, 우리도 데이터셋을 표준화(즉 전체 합이 1이 되고 비율은 동일하게)하기 위해 전체 합을 구합시다.
        n = 0 #freq 데이터가 dataframe 구조라서, 그 인덱스를 불러오기 위해 n이라는 변수를 쓰겠습니다.
        for letter in freq["letter"]: #이제부터 for 루프를 통해 글자가 같은 빈도수의 차를 구할 것입니다. 처음에는 일반 빈도 통계의 글자를 하나 꺼냅니다.
            for letter2 in self.data.keys(): #위에서 꺼낸 글자는 이제 고정되고, 이제는 우리가 모은 데이터의 글자의 루프를 돌립시다.
                                            #이 과정은 알파벳 이외의 글자를 필터링하는 역할도 겸합니다.
                if letter == letter2: #이 조건은 오직 두 데이터셋의 키인 알파벳이 같을 때밖에 없습니다. 앞서 말한 필터링이 이 논리 검사로 인해 가능합니다.



                                                                                                 # 객체의 데이터의 통계가 입력됩니다.
                    aList.append((self.data[letter2]) / sum) #초기 프로그래밍 코딩 이후에, 약간 조정하여 자신의 데이터를 기록하는 리스를 만들었습니다.
                    bList.append(freq.loc[n,"frequency"])


            n += 1

        self.sortedData = np.array(aList) #이 코드는 첫 예제들에서 조금 더 코드를 간략화한 경우입니다. 데이터 처리도 유연해질 수 있습니다.
        self.standardData = np.array(bList) #이 애트리뷰트에 데이터들을 집어넣음으로서, 우리가 원하는대로 데이터를 처리하는 것이 가능해집니다.

        return self.sortedData