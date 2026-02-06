from flask import *
import sqlite3

app= Flask(__name__)
@app.route("/")
def index(): 
    return render_template("sindex.html")
@app.route("/add")
def add():
    return render_template("sadd.html")
@app.route("/savedetails",methods = ["POST","GET"])
def savedeatils():
    msg = "msg"
    if request.method == "POST":
        try:
            name = request.form["name"]
            email = request.form["email"]
            address = request.form["address"]
            number = request.form["number"]
            college_name = request.form["college_name"]
            city = request.form["city"]
            state = request.form["state"]
            with sqlite3.connect("senroll.db") as con:
                cur = con.cursor()
                cur.execute("INSERT into ens (name,email,address,number,college_name,city,state) values (?,?,?,?,?,?,?)",(name,email,address,number,college_name,city,state))
                con.commit()
                msg = "Your Details have been Successfully Submitted"
        except:
            con.rollback()
            msg = "Sorry! Please fill all the details in the form"
        finally:
            return render_template("ssuccess.html",msg = msg)
            con.close()
            
@app.route("/view")
def view():
    con = sqlite3.connect("senroll.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from ens")
    rows = cur.fetchall()
    return render_template("sview.html",rows = rows)
    
@app.route("/data")
def data_response():
    con = sqlite3.connect("senroll.db")
    cur = con.cursor()
    cur.execute("select * from ens")
    rows = cur.fetchall()
    js = jsonify({'students':rows})
    return js
    
if __name__ == "__main__":
    app.run(debug = True,port=1234)