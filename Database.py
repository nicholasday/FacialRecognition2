import requests
from User import User

URL = 'http://signin-api.nickendo.com'

class Database():
    users = []

    def signin(self, user):
        pass
    
    def signout(self, user):
        pass

    def loadSignins(self, user):
        pass

    def getMembers(self):
        r = requests.get(URL+'/members')
        json = r.json()

        for user in json['members']:
            self.users.append(User(user['name'], int(user['id']), user['last_pizza'], int(user['team'])))

        for user in self.users:
            print(user)