from flask import Flask,jsonify,render_template

obj = Flask('__main__')

@obj.route("/")
def f1():
	return "<h1>Welcome to Flask App</h1>"

@obj.route("/aboutus")
def f2():
	return "<h2>This is About us page</h2>"

@obj.route("/mydata")
def f3():
	d={'model':['gpt4.0','gpt5.0','lamma']}
	return jsonify(d)

@obj.route("/mypage")
def f4():
	return render_template('response.html')


if __name__ == '__main__':
	obj.run(debug=True,port=3234) # default port 5000 ;