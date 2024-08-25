import express from 'express'
import pg from 'pg';
import bodyParser from 'body-parser';
import axios from 'axios';
import dotenv from 'dotenv'

const PORT =5000;
dotenv.config();
const db = new pg.Client({
  user: process.env.user,
  host: process.env.host,
  database: process.env.database,
  password: process.env.password,
  port: process.env.port,
});

db.connect();

let chat;
let chatname;
let chatnames;
let curr_chatid;


const app = express();
app.use(express.static("public"));
app.use(bodyParser.urlencoded({extended:"true"}));

app.get("/",async(req,res)=>{
  chatnames = await db.query("select * from chat");
  if(chatnames.rows.length===0){
    chatnames = undefined;
  }
  res.render("index.ejs", {
    chatnames:chatnames ? chatnames.rows : chatnames,
    chatname:undefined,
    chat:undefined,
  });
})

app.get("/chats",async (req,res)=>{
  const {chatid} = req.query;
  curr_chatid = chatid;
  chat = await db.query("select * from chatmesages where chatid=$1",[chatid]);
  if(chat.rows.length==0){
    chat = undefined;
  }
  res.render("index.ejs", {
    chatnames:chatnames ? chatnames.rows : chatnames,
    chatname: chatname ? chatname : undefined,
    chat: chat ? chat.rows : undefined,
  });
})

app.post("/newchat" , async (req,res)=>{
  const {chatname} = req.body;
  try {
    await db.query("insert into chat (chatname) values ($1)",[chatname]);
    res.redirect("/");
  } catch (error) {
    res.send({error:error});
  }

}
)

app.post("/chat" , async (req,res)=>{
  const chatid = curr_chatid;
  const message = req.body.message;
  try{
    const response = await axios.post("https://gbrh7rr7-5000.inc1.devtunnels.ms/",{
      question:message
    });
    await db.query("insert into chatmesages (message,response,chatid) values ($1,$2,$3)",[message,response.data.response,chatid]);
    res.redirect(`/chats?chatid=${chatid}`);
  }catch(error){
    res.send({error:error});
  }
})

app.listen(PORT,()=>console.log('Server is running'));
