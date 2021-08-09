package main

import (
	//"net/url"
	"net/http"
	"fmt"
	"os"
	"encoding/json"
	"strings"
	"io/ioutil"
	// "bytes// "
	// "strconv"
	"github.com/PuerkitoBio/goquery"
)

type Ncode struct {
	Ncode string `json:"ncode"`
}

type Info struct {
	Title string `json:"title"`
	Seq string `json:"sequence"`
}

func GetPage(url string, info []Info) []Info {
	fmt.Println(url)
	doc, _ := goquery.NewDocument(url)

	var ins Info 
	doc.Find(".novel_title").Each(func(_ int, s *goquery.Selection) {
		ins.Title = s.Text()
	})
	
	doc.Find("#novel_honbun").Each(func(_ int, s *goquery.Selection) {
		ins.Seq = s.Text()
	})

	info = append(info, ins)
	return info
}

func main() {
	c_url := "https://api.syosetu.com/novelapi/api/"
	
	req, err := http.NewRequest( "GET", c_url, nil )
	
	if err != nil {
		fmt.Println( "Error:http" )
		fmt.Println( err )
		os.Exit( 0 )
	}

	values := req.URL.Query()
	values.Add( "genre", "101" )
	values.Add( "keyword", "婚約破棄" )
	values.Add( "type", "t" )
	values.Add( "out", "json" )
	values.Add( "of", "n" )
	values.Add( "lim", "1" )
	values.Add( "order", "yearlypoint" )
    // 戻す
    req.URL.RawQuery = values.Encode()

	req.Header.Set("Content-Type", "application/json")

	client := new( http.Client )
	
	resp, err := client.Do( req )

	var id []Ncode 
	if resp != nil {
		defer resp.Body.Close()
		var byteArray, _ = ioutil.ReadAll( resp.Body )
		_ = json.Unmarshal(byteArray, &id)
	} else {
		fmt.Println( err )
		os.Exit( 0 )
	}

	var info []Info
	c_url = "https://ncode.syosetu.com/"
	for _, code := range id {
		if code.Ncode != "" {
			info = GetPage(c_url+strings.ToLower(code.Ncode), info)
		}
	}

	f, err := os.Create("save/data_output.json")
	if err != nil {
		fmt.Println( err )
		os.Exit( 0 )
	}
	defer f.Close()
	
	err = json.NewEncoder(f).Encode(info)
	if err != nil {
		fmt.Println( err )
		os.Exit( 0 )
	}
}
